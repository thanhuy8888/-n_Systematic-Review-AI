from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict, Any
import tempfile
import os
from pydantic import BaseModel

from apps.api.database import get_db
from apps.api.models import Paper, ScreeningLabel, ReviewAction
from sr_core.screening_model.transformer_screen import get_screening_model
from sr_core.extraction.extractive_qa import get_qa_pipeline

router = APIRouter(
    prefix="/screening",
    tags=["screening"]
)

class ReviewCriteria(BaseModel):
    population: str
    intervention: str
    comparison: str
    outcome: str
    studyType: str

class PaperSchema(BaseModel):
    id: str
    title: str
    abstract: str
    fullText: Optional[str] = None
    
class PredictRequest(BaseModel):
    paper: PaperSchema
    criteria: ReviewCriteria

class ExtractRequest(BaseModel):
    paper: PaperSchema

@router.get("/")
def get_screening_board_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    ai_label: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Fetches papers for the screening board.
    Optionally filters by the AI label ('include', 'exclude', 'uncertain').
    """
    query = db.query(Paper)
    
    # Simple join if ai_label filter is active
    if ai_label:
        query = query.join(ScreeningLabel).filter(ScreeningLabel.ai_label == ai_label)
        
    # Get total count (for pagination)
    total = query.count()
    
    # Get paginated papers
    papers = query.order_by(desc(Paper.year)).offset(skip).limit(limit).all()
    
    # Format response
    formatted_papers = []
    for paper in papers:
        # Get AI label (if exists)
        label_record = db.query(ScreeningLabel).filter(ScreeningLabel.paper_id == paper.paper_id).first()
        
        action_record = db.query(ReviewAction).filter(
            ReviewAction.paper_id == paper.paper_id
        ).order_by(desc(ReviewAction.timestamp)).first()
        
        formatted_papers.append({
            "id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "year": paper.year,
            "journal": paper.journal,
            "aiLabel": label_record.ai_label if label_record else "uncertain",
            "aiScore": label_record.ai_score if label_record else 0.5,
            "reviewerDecision": action_record.action if action_record else None
        })
        
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": formatted_papers
    }

@router.post("/{paper_id}/action")
def submit_review_action(
    paper_id: str,
    action: str = Query(..., description="'include', 'exclude', or 'uncertain'"),
    notes: Optional[str] = None,
    user_id: str = "default_user",
    db: Session = Depends(get_db)
):
    """
    Submits a final human decision for a specific paper.
    """
    valid_actions = ["include", "exclude", "uncertain"]
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of {valid_actions}")
        
    # Verify paper exists
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
        
    # Record action
    review_action = ReviewAction(
        paper_id=paper_id,
        reviewer_id=user_id,
        action=action,
        comment=notes
    )
    db.add(review_action)
    db.commit()
    
    return {"status": "success", "action": action, "paper_id": paper_id}

@router.post("/predict")
def predict_screening(request: PredictRequest):
    """
    Evaluates the paper against criteria using the PyTorch Transformer pipeline.
    For full-text screening: if fullText is present, use it instead of abstract.
    """
    paper = request.paper
    criteria = dict(request.criteria)
    
    # 1. Load singleton Transformer pipeline
    model = get_screening_model()
    
    # 2. Determine which text to screen against
    # If fullText is provided (PDF), use it for deeper analysis
    text_to_screen = paper.abstract
    if paper.fullText and len(paper.fullText.strip()) > 200:
        # Use first 3000 chars of fullText for full-text screening
        text_to_screen = paper.fullText[:3000]
        print(f"[Predict] Using fullText ({len(paper.fullText)} chars) for: {paper.title[:50]}")
    else:
        print(f"[Predict] Using abstract ({len(paper.abstract)} chars) for: {paper.title[:50]}")
    
    # 3. Run Criteria-Aware evaluation
    result = model.predict(title=paper.title, abstract=text_to_screen, criteria=criteria)
    
    return {
         "decision": result["decision"],
         "confidence": result["confidence"],
         "is_uncertain": result["is_uncertain"],
         "reason": result["reason"]
    }

def active_learning_update():
    """Background task to simulate or eventually run adapter fine-tuning."""
    print("Running Active Learning update in background...")
    # TODO: Load ReviewAction logs, format to Dataset, run HuggingFace Trainer

@router.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Triggers the Active Learning loop to fine-tune the Transformer 
    based on the latest user overrides and labels.
    """
    background_tasks.add_task(active_learning_update)
    return {"status": "success", "message": "Active Learning retraining triggered in background."}

@router.post("/extract")
def extract_data(request: ExtractRequest):
    """
    Extracts structured data from paper using RoBERTa Extractive QA.
    """
    import re
    paper = request.paper
    qa_pipeline = get_qa_pipeline()
    
    # Prioritize passing the complete fullText to the RoBERTa model if it was a PDF upload
    text_to_analyze = paper.fullText if paper.fullText else (paper.abstract if paper.abstract else "")
    
    # Clean up raw PDF text: remove excessive line breaks, page numbers, headers
    text_to_analyze = re.sub(r'\n{2,}', '\n', text_to_analyze)
    text_to_analyze = re.sub(r'\s{3,}', ' ', text_to_analyze)
    text_to_analyze = text_to_analyze.strip()
    
    print(f"[Extract] Text length: {len(text_to_analyze)} chars for paper: {paper.title[:60]}")
    print(f"[Extract] First 200 chars: {text_to_analyze[:200]}")
    
    if qa_pipeline.model is None:
         # Fallback mock for UI usage without downloading weights
         return {
             "methodology": "Mock RCT (HuggingFace weights missing)",
             "sampleSize": "N=200",
             "keyFindings": "Model was not loaded locally. See server logs.",
             "limitations": "Placeholder limitations",
             "riskOfBias": "High"
         }
         
    extracted = qa_pipeline.extract_structured_data(paper.title, text_to_analyze)
    print(f"[Extract] Results: {extracted}")
    return extracted

@router.post("/parse_pdf")
async def parse_pdf_file(file: UploadFile = File(...)):
    """
    Accepts a raw PDF upload from the React UI and uses PyMuPDF 
    to extract metadata + full text for screening/extraction.
    """
    import traceback
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        from sr_core.ingest.pdf_parser import extract_text_from_pdf
        result = extract_text_from_pdf(tmp_path)
        
        if isinstance(result, dict):
            if "error" in result:
                return {"text": "", "title": "", "authors": "", "abstract": "", "doi": "", "year": "", "keywords": [], "error": result["error"]}
            return {
                "text": result.get("raw_text", ""),
                "title": result.get("title", ""),
                "authors": result.get("authors", ""),
                "abstract": result.get("abstract", ""),
                "doi": result.get("doi", ""),
                "year": result.get("year", ""),
                "keywords": result.get("keywords", []),
            }
        else:
            return {"text": str(result), "title": "", "authors": "", "abstract": "", "doi": "", "year": "", "keywords": []}
            
    except Exception as e:
        print(f"[parse_pdf] CRITICAL ERROR: {e}")
        traceback.print_exc()
        return {"text": "", "title": "", "authors": "", "abstract": "", "doi": "", "year": "", "keywords": [], "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.delete("/{paper_id}")
def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """
    Deletes a specific paper and its dependent records from the database.
    """
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
        
    # Delete associated records manually (or let cascade handle it if configured, but let's be explicit)
    db.query(ScreeningLabel).filter(ScreeningLabel.paper_id == paper_id).delete()
    db.query(ReviewAction).filter(ReviewAction.paper_id == paper_id).delete()
    
    db.delete(paper)
    db.commit()
    
    return {"status": "success", "message": f"Deleted paper {paper_id}"}

