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


# ---------------------------------------------------------------------------
# Adaptive (structure-aware) chunking
# ---------------------------------------------------------------------------

# IMRaD + common biomedical headings. Match line-start, case-insensitive,
# optionally numbered ("2. Methods", "II. Results").
_SECTION_HEADERS = [
    "abstract", "summary",
    "introduction", "background",
    "materials and methods", "methods", "methodology",
    "experimental procedures", "experimental design",
    "results", "findings",
    "discussion", "conclusion", "conclusions",
    "limitations", "future work",
    "references", "acknowledgments", "acknowledgements",
]

_HEADER_RE = re.compile(
    r"^\s*(?:(?:[0-9IVX]+)[\.\)]\s*)?(" + "|".join(_SECTION_HEADERS) + r")\s*[:.\n]",
    re.IGNORECASE | re.MULTILINE,
)


def split_into_sections(text: str) -> list:
    """
    Adaptive segmentation: split a paper into IMRaD-style sections instead
    of fixed-size token windows. Returns:
        [{"section": "methods", "text": "..."}]
    Falls back to a single "body" section if no headers are found.
    """
    if not text:
        return []

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [{"section": "body", "text": text.strip()}]

    sections = []
    # Preamble before first header (often title/authors block)
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append({"section": "preamble", "text": pre})

    for i, m in enumerate(matches):
        name = m.group(1).lower().strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append({"section": name, "text": body})
    return sections


def _paragraphs(text: str) -> list:
    """Split a section into paragraphs (blank-line delimited)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_by_structure(
    text: str,
    max_chars: int = 2500,
    overlap_chars: int = 300,
) -> list:
    """
    Structure-aware chunker.

    Strategy (per Chapter 4 §1.2 recommendation: "adaptive chunking methods
    to improve segmentation boundaries based on the structure of the
    documents rather than the size of the tokens"):

    1. Split by IMRaD section headers.
    2. Within each section, accumulate paragraphs up to `max_chars`.
    3. Carry a small `overlap_chars` tail from the previous chunk so
       cross-paragraph relationships (the boundary-sensitive extraction
       problem flagged in §1.5) are preserved.

    Each returned chunk carries provenance: which section it came from,
    its order, and char offsets — so downstream evidence grounding can
    cite section + offsets, not just text.
    """
    if not text:
        return []

    chunks = []
    chunk_idx = 0

    for sec in split_into_sections(text):
        section_name = sec["section"]
        paras = _paragraphs(sec["text"])

        buf = ""
        tail = ""  # overlap from previous chunk in this section

        def flush():
            nonlocal buf, tail, chunk_idx
            if not buf.strip():
                return
            content = (tail + " " + buf).strip() if tail else buf.strip()
            chunks.append({
                "chunk_id": chunk_idx,
                "section": section_name,
                "text": content,
                "char_len": len(content),
            })
            chunk_idx += 1
            tail = content[-overlap_chars:] if overlap_chars > 0 else ""
            buf = ""

        for para in paras:
            # Paragraph itself bigger than the window — hard-split it.
            if len(para) > max_chars:
                # Flush whatever was buffered first
                if buf:
                    flush()
                for start in range(0, len(para), max_chars - overlap_chars):
                    piece = para[start : start + max_chars]
                    chunks.append({
                        "chunk_id": chunk_idx,
                        "section": section_name,
                        "text": piece,
                        "char_len": len(piece),
                    })
                    chunk_idx += 1
                tail = para[-overlap_chars:] if overlap_chars > 0 else ""
                continue

            if len(buf) + len(para) + 1 > max_chars:
                flush()
            buf = (buf + "\n\n" + para) if buf else para

        if buf:
            flush()

    return chunks


def flatten_tables(tables: list) -> str:
    """
    Convert PyMuPDF-extracted tables (list of {page, rows}) to a single
    pipe-delimited text block that can be injected into the QA context.
    This keeps tabular outcomes (lipid panels, body weights) visible to
    the extractive model instead of letting them disappear into the
    prose noise that the report flagged.
    """
    if not tables:
        return ""
    out = []
    for t in tables:
        out.append(f"[Table p.{t.get('page', '?')}]")
        for row in t.get("rows", []):
            cells = [str(c).strip() if c is not None else "" for c in row]
            if any(cells):
                out.append(" | ".join(cells))
        out.append("")
    return "\n".join(out)

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
