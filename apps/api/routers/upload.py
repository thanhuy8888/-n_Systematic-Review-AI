import os
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from apps.api.database import get_db
from apps.api.models import Paper, ScreeningLabel
from sr_core.ingest.parser import parse_ris, parse_bib
from sr_core.ingest.pdf_parser import extract_text_from_pdf

router = APIRouter(
    prefix="/upload",
    tags=["upload"]
)

UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_files(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    Endpoint to upload files (.ris, .bib, .pdf).
    Saves files locally and triggers ingestion into the DB.
    """
    results = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ext = os.path.splitext(file.filename)[1].lower()
        
        papers_added = 0
        try:
            if ext == ".ris":
                papers = parse_ris(file_path)
                papers_added = _insert_papers(db, papers)
            elif ext == ".bib":
                papers = parse_bib(file_path)
                papers_added = _insert_papers(db, papers)
            elif ext == ".pdf":
                # PDF typically represents a single paper or full text
                parsed = extract_text_from_pdf(file_path)
                if isinstance(parsed, dict) and "error" not in parsed:
                    raw_text = parsed.get("raw_text", "")
                    abstract_text = parsed.get("abstract", "") or raw_text[:2000]
                    paper = Paper(
                        paper_id=parsed.get("paper_id") or f"pdf_{file.filename}", 
                        title=parsed.get("title") or file.filename,
                        abstract=abstract_text,
                        authors=parsed.get("authors"),
                        year=int(parsed.get("year")) if str(parsed.get("year")).isdigit() else None,
                        doi=parsed.get("doi"),
                        journal=parsed.get("journal"),
                        ingest_status="pdf_parsed"
                    )
                else:
                    error_msg = parsed.get("error", "Unknown parsing error") if isinstance(parsed, dict) else str(parsed)
                    paper = Paper(
                        paper_id=f"pdf_{file.filename}", 
                        title=file.filename,
                        abstract=f"Error parsing PDF: {error_msg}",
                        ingest_status="error"
                    )
                db.merge(paper)
                db.commit()
                papers_added = 1
            else:
                 results.append({"filename": file.filename, "status": "failed", "reason": "Unsupported extension"})
                 continue
                 
            results.append({"filename": file.filename, "status": "success", "papers_added": papers_added})
            
        except Exception as e:
             results.append({"filename": file.filename, "status": "failed", "reason": str(e)})

    return {"results": results}

def _insert_papers(db: Session, papers_data: list):
    count = 0
    for p_data in papers_data:
        # Check if exists
        existing = db.query(Paper).filter(Paper.paper_id == p_data['paper_id']).first()
        if not existing:
            paper = Paper(
                paper_id=p_data['paper_id'],
                title=p_data.get('title'),
                abstract=p_data.get('abstract'),
                authors=p_data.get('authors'),
                year=p_data.get('year'),
                journal=p_data.get('journal'),
                doi=p_data.get('doi'),
                url=p_data.get('url'),
                ingest_status=p_data.get('ingest_status', 'pending')
            )
            db.add(paper)
            count += 1
            
            # If human_label came from Rayyan data (though not typical via standard upload)
            if 'human_label' in p_data:
                label = ScreeningLabel(
                    paper_id=p_data['paper_id'],
                    ai_label=p_data['human_label'], # Treating ground truth as AI label placeholder for now
                    ai_score=1.0,
                    rationale_json='{"source": "ground_truth"}'
                )
                db.add(label)
                
    db.commit()
    return count
