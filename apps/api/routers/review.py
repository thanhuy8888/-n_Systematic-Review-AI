"""
Human-in-the-loop (HITL) review endpoints.

Implements the workflow recommended in Chapter 4 §1.2 of the thesis:
domain experts validate / override AI extractions, especially the
low-confidence ones that the QA model flagged.

Endpoints:
    POST   /review/extractions/persist  — persist a fresh extraction batch
    GET    /review/extractions          — list (filter by paper / status / low_confidence)
    GET    /review/extractions/{id}     — single record
    PATCH  /review/extractions/{id}     — submit human correction
    POST   /review/extractions/{id}/accept — accept AI value as-is
    GET    /review/queue                — papers with at least one pending low-conf field
    GET    /review/stats                — overall HITL progress
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from apps.api.database import get_db
from apps.api.models import Paper, Extraction
from sr_core.extraction.extractive_qa import CONFIDENCE_REVIEW_THRESHOLD

router = APIRouter(prefix="/review", tags=["review"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExtractionFieldIn(BaseModel):
    field: str
    value: str
    ai_value: Optional[str] = None
    confidence: float = 0.0
    evidence_section: Optional[str] = None
    evidence_span: Optional[str] = None


class PersistRequest(BaseModel):
    paper_id: str
    model_version: Optional[str] = "roberta-base-squad2"
    fields: List[ExtractionFieldIn]


class CorrectionIn(BaseModel):
    value: str = Field(..., description="The human-corrected value")
    reviewer_id: Optional[str] = "default_user"
    comment: Optional[str] = None


class AcceptIn(BaseModel):
    reviewer_id: Optional[str] = "default_user"
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@router.post("/extractions/persist")
def persist_extraction(payload: PersistRequest, db: Session = Depends(get_db)):
    """
    Persist a batch of extraction results for a paper. Existing pending
    rows for the same (paper_id, field) are overwritten; rows already
    reviewed by a human are left alone (we never silently overwrite
    expert corrections).
    """
    paper = db.query(Paper).filter(Paper.paper_id == payload.paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    saved = 0
    skipped_reviewed = 0
    for f in payload.fields:
        existing = (
            db.query(Extraction)
            .filter(Extraction.paper_id == payload.paper_id, Extraction.field == f.field)
            .first()
        )
        if existing and existing.status in ("accepted", "corrected"):
            # Don't clobber a human decision
            skipped_reviewed += 1
            continue

        if existing:
            existing.value = f.value
            existing.ai_value = f.ai_value or f.value
            existing.confidence = f.confidence
            existing.evidence_section = f.evidence_section
            existing.evidence_span = f.evidence_span
            existing.model_version = payload.model_version
            existing.status = "pending"
        else:
            db.add(Extraction(
                paper_id=payload.paper_id,
                field=f.field,
                value=f.value,
                ai_value=f.ai_value or f.value,
                confidence=f.confidence,
                evidence_section=f.evidence_section,
                evidence_span=f.evidence_span,
                model_version=payload.model_version,
                status="pending",
            ))
        saved += 1

    db.commit()
    return {
        "status": "success",
        "paper_id": payload.paper_id,
        "saved": saved,
        "skipped_already_reviewed": skipped_reviewed,
    }


# ---------------------------------------------------------------------------
# Listing / fetching
# ---------------------------------------------------------------------------

def _serialize(ex: Extraction) -> Dict[str, Any]:
    return {
        "id": ex.id,
        "paper_id": ex.paper_id,
        "field": ex.field,
        "value": ex.value,
        "ai_value": ex.ai_value,
        "confidence": ex.confidence,
        "evidence_section": ex.evidence_section,
        "evidence_span": ex.evidence_span,
        "model_version": ex.model_version,
        "status": ex.status,
        "reviewer_id": ex.reviewer_id,
        "reviewer_comment": ex.reviewer_comment,
        "low_confidence": (ex.confidence or 0) < CONFIDENCE_REVIEW_THRESHOLD,
        "created_at": ex.created_at.isoformat() if ex.created_at else None,
        "reviewed_at": ex.reviewed_at.isoformat() if ex.reviewed_at else None,
    }


@router.get("/extractions")
def list_extractions(
    paper_id: Optional[str] = None,
    status: Optional[str] = Query(None, description="pending|accepted|corrected|rejected"),
    field: Optional[str] = None,
    low_confidence_only: bool = False,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(Extraction)
    if paper_id:
        q = q.filter(Extraction.paper_id == paper_id)
    if status:
        q = q.filter(Extraction.status == status)
    if field:
        q = q.filter(Extraction.field == field)
    if low_confidence_only:
        q = q.filter(Extraction.confidence < CONFIDENCE_REVIEW_THRESHOLD)

    total = q.count()
    rows = (
        q.order_by(Extraction.confidence.asc(), desc(Extraction.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": [_serialize(r) for r in rows],
    }


@router.get("/extractions/{extraction_id}")
def get_extraction(extraction_id: int, db: Session = Depends(get_db)):
    ex = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if not ex:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return _serialize(ex)


# ---------------------------------------------------------------------------
# Human actions
# ---------------------------------------------------------------------------

@router.patch("/extractions/{extraction_id}")
def submit_correction(
    extraction_id: int,
    body: CorrectionIn,
    db: Session = Depends(get_db),
):
    """Reviewer overrides the AI value."""
    ex = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if not ex:
        raise HTTPException(status_code=404, detail="Extraction not found")

    ex.value = body.value
    ex.status = "corrected"
    ex.reviewer_id = body.reviewer_id
    ex.reviewer_comment = body.comment
    ex.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(ex)
    return {"status": "success", "extraction": _serialize(ex)}


@router.post("/extractions/{extraction_id}/accept")
def accept_extraction(
    extraction_id: int,
    body: AcceptIn,
    db: Session = Depends(get_db),
):
    """Reviewer confirms the AI value as correct (no edit)."""
    ex = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if not ex:
        raise HTTPException(status_code=404, detail="Extraction not found")

    ex.status = "accepted"
    ex.reviewer_id = body.reviewer_id
    ex.reviewer_comment = body.comment
    ex.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(ex)
    return {"status": "success", "extraction": _serialize(ex)}


@router.post("/extractions/{extraction_id}/reject")
def reject_extraction(
    extraction_id: int,
    body: AcceptIn,
    db: Session = Depends(get_db),
):
    """Reviewer marks the AI value as wrong with no replacement."""
    ex = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if not ex:
        raise HTTPException(status_code=404, detail="Extraction not found")

    ex.status = "rejected"
    ex.reviewer_id = body.reviewer_id
    ex.reviewer_comment = body.comment
    ex.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(ex)
    return {"status": "success", "extraction": _serialize(ex)}


# ---------------------------------------------------------------------------
# Queue / stats
# ---------------------------------------------------------------------------

@router.get("/queue")
def review_queue(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Lists papers that have at least one pending low-confidence extraction.
    Ordered by # of fields needing review (descending) so reviewers tackle
    the most-uncertain papers first.
    """
    pending_counts = (
        db.query(
            Extraction.paper_id,
            func.count(Extraction.id).label("n_pending"),
        )
        .filter(
            Extraction.status == "pending",
            Extraction.confidence < CONFIDENCE_REVIEW_THRESHOLD,
        )
        .group_by(Extraction.paper_id)
        .order_by(desc("n_pending"))
        .offset(skip)
        .limit(limit)
        .all()
    )

    out = []
    for paper_id, n_pending in pending_counts:
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            continue
        out.append({
            "paper_id": paper_id,
            "title": paper.title,
            "year": paper.year,
            "pending_low_conf_fields": n_pending,
        })
    return {"data": out, "skip": skip, "limit": limit}


@router.get("/stats")
def review_stats(db: Session = Depends(get_db)):
    """HITL progress dashboard."""
    by_status = dict(
        db.query(Extraction.status, func.count(Extraction.id))
        .group_by(Extraction.status)
        .all()
    )
    total = sum(by_status.values())
    low_conf_pending = (
        db.query(func.count(Extraction.id))
        .filter(
            Extraction.status == "pending",
            Extraction.confidence < CONFIDENCE_REVIEW_THRESHOLD,
        )
        .scalar() or 0
    )
    avg_conf = db.query(func.avg(Extraction.confidence)).scalar() or 0.0
    reviewed = by_status.get("accepted", 0) + by_status.get("corrected", 0)
    return {
        "total_extractions": total,
        "by_status": by_status,
        "low_confidence_pending": low_conf_pending,
        "average_confidence": round(float(avg_conf), 4),
        "reviewed_pct": round(100 * reviewed / total, 1) if total else 0.0,
        "threshold": CONFIDENCE_REVIEW_THRESHOLD,
    }
