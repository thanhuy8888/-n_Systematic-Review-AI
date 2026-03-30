import fitz # PyMuPDF
import re

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extracts text AND structured metadata (title, authors, abstract, DOI, year) from a PDF.
    Uses a combination of PyMuPDF metadata API + regex heuristics on the first pages.
    """
    text_content = ""
    metadata = {}
    
    try:
        doc = fitz.open(pdf_path)
        
        # 1. Extract built-in PDF metadata (many academic PDFs embed this)
        pdf_meta = doc.metadata or {}
        metadata["pdf_title"] = pdf_meta.get("title", "").strip()
        metadata["pdf_author"] = pdf_meta.get("author", "").strip()
        metadata["pdf_subject"] = pdf_meta.get("subject", "").strip()
        
        # 2. Extract all text
        for page in doc:
            text_content += page.get_text() + "\n"
        
        # 3. Extract title from first page (largest font text)
        first_page = doc[0]
        blocks = first_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        title_candidates = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > 13 and len(span["text"].strip()) > 5:
                            title_candidates.append((span["size"], span["text"].strip()))
        
        # Sort by font size descending, largest = likely title
        title_candidates.sort(key=lambda x: x[0], reverse=True)
        
        doc.close()
    except Exception as e:
        return {"error": str(e)}

    # ---- Heuristic metadata extraction ----
    first_2000 = text_content[:3000]
    
    # Title: prefer PDF metadata, then largest-font text, then first non-empty line
    title = metadata.get("pdf_title", "")
    if not title and title_candidates:
        title = title_candidates[0][1]
    if not title:
        for line in first_2000.split("\n"):
            line = line.strip()
            if len(line) > 15 and not line.lower().startswith(("http", "doi", "©", "copyright")):
                title = line
                break
    
    # Authors: prefer PDF metadata, then heuristic (lines near top with commas/ands)
    authors = metadata.get("pdf_author", "")
    if not authors:
        author_match = re.search(
            r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)*)',
            first_2000
        )
        if author_match:
            authors = author_match.group(1).strip()
    
    # Abstract: search for "Abstract" heading
    abstract = ""
    abstract_match = re.search(
        r'(?i)\babstract\b\s*[:\-\n]\s*(.*?)(?=\n\s*\n\s*(?:[A-Z][a-z]+\s|1\.\s|Introduction|Background|Keywords|Key\s*words))',
        first_2000, re.DOTALL
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)  # normalize whitespace
    
    # DOI
    doi = ""
    doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', text_content[:5000])
    if doi_match:
        doi = doi_match.group(1).rstrip(".,;)")
    
    # Year
    year = ""
    year_match = re.search(r'\b((?:19|20)\d{2})\b', first_2000)
    if year_match:
        year = year_match.group(1)
    
    # Keywords
    keywords = []
    kw_match = re.search(r'(?i)(?:keywords?|key\s*words?)\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z])', first_2000, re.DOTALL)
    if kw_match:
        kw_text = kw_match.group(1).strip()
        keywords = [k.strip() for k in re.split(r'[;,·•]', kw_text) if k.strip()]

    return {
        "raw_text": text_content,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "year": year,
        "keywords": keywords,
        "needs_ocr": len(text_content.strip()) < 50
    }
