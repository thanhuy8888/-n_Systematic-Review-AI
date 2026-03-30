import os
import hashlib
import re
import rispy
import bibtexparser
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """Basic text normalization: HTML stripping, lowercase, extra spaces."""
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_paper_id(doi: str, title: str, year: str, authors: str) -> str:
    """Generate a stable paper ID based on DOI or Title + Year + First Author."""
    if doi:
        base_str = doi.strip().lower()
    else:
        first_author = authors.split(',')[0].strip().lower() if authors else ""
        year_str = str(year).strip() if year else ""
        title_str = title.strip().lower() if title else ""
        base_str = f"{title_str}|{year_str}|{first_author}"
    
    return hashlib.sha256(base_str.encode('utf-8')).hexdigest()

def parse_ris(file_path: str) -> list:
    """Parse a RIS file and return a list of standard paper dicts."""
    parsed_papers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        entries = rispy.load(f)
        for entry in entries:
            title = entry.get('title', '') or entry.get('primary_title', '')
            abstract = entry.get('abstract', '')
            doi = entry.get('doi', '')
            year = entry.get('year', '')
            authors_list = entry.get('authors', [])
            authors = ', '.join(authors_list)
            journal = entry.get('journal_name', '')

            paper_id = generate_paper_id(doi, title, year, authors)
            
            parsed_papers.append({
                "paper_id": paper_id,
                "title": clean_text(title),
                "abstract": clean_text(abstract),
                "keywords": "", # Could extract if present
                "authors": authors,
                "year": int(year) if str(year).isdigit() else None,
                "doi": doi,
                "journal": journal,
                "source_path": file_path,
                "ingest_status": "ok" if abstract else "missing_abstract"
            })
    return parsed_papers

def parse_bib(file_path: str) -> list:
    """Parse a BIB (BibTeX) file and return a list of standard paper dicts."""
    parsed_papers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
        for entry in bib_database.entries:
            title = entry.get('title', '')
            abstract = entry.get('abstract', '')
            doi = entry.get('doi', '')
            year = entry.get('year', '')
            authors = entry.get('author', '').replace(' and ', ', ')
            journal = entry.get('journal', '')

            paper_id = generate_paper_id(doi, title, year, authors)
            
            parsed_papers.append({
                "paper_id": paper_id,
                "title": clean_text(title),
                "abstract": clean_text(abstract),
                "keywords": "",
                "authors": authors,
                "year": int(year) if str(year).isdigit() else None,
                "doi": doi,
                "journal": journal,
                "source_path": file_path,
                "ingest_status": "ok" if abstract else "missing_abstract"
            })
    return parsed_papers
