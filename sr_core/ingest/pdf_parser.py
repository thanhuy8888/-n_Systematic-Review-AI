"""
PDF parser with OCR fallback and post-processing.

Improvements vs. original (per Chapter 4 recommendations of the thesis):
1. OCR fallback (pytesseract) when PyMuPDF returns near-empty text
   (handles scanned / legacy biomedical PDFs).
2. OCR post-processing: dehyphenation across line breaks, ligature
   normalization, whitespace cleanup — reduces artifacts that previously
   propagated into screening/extraction.
3. Section-aware text capture (page_texts) preserved so the downstream
   chunker can do structure-aware splitting (see parser.chunk_by_structure).
4. Table extraction via PyMuPDF find_tables() so tabular outcomes are
   not flattened into prose noise.

OCR and table extraction are *best-effort* — if the optional dependency
(pytesseract / Pillow) is unavailable, the parser still returns text and
flags `needs_ocr=True` so the caller can decide.
"""
import fitz  # PyMuPDF
import re
import unicodedata

# Optional OCR deps — imported lazily so the package works without them.
try:
    import pytesseract
    from PIL import Image
    import io
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# OCR post-processing
# ---------------------------------------------------------------------------

# Common ligatures that OCR / PDF extraction leaves behind.
_LIGATURE_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st",
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    " ": " ",
}

def _normalize_text(text: str) -> str:
    """Unicode normalize + replace common ligatures / smart punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURE_MAP.items():
        text = text.replace(src, dst)
    return text


def _dehyphenate(text: str) -> str:
    """
    Join words broken across line breaks: 'hyper-\nglycemia' -> 'hyperglycemia'.
    Also collapses 3+ blank lines into a paragraph break.
    """
    if not text:
        return ""
    # word-\nword  ->  wordword   (academic PDFs split words at line end)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Single newline inside a sentence (no period before) -> space
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)
    # 3+ newlines -> 2 (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clean_ocr_artifacts(text: str) -> str:
    """Common OCR / extraction noise reduction."""
    if not text:
        return ""
    # Repeated single-character "l l l" noise from page rulers
    text = re.sub(r"(?:\s[a-zA-Z0-9]){10,}\s", " ", text)
    # Repeated form-feed / control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Trim trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text


def _post_process(text: str) -> str:
    text = _normalize_text(text)
    text = _dehyphenate(text)
    text = _clean_ocr_artifacts(text)
    return text


# ---------------------------------------------------------------------------
# OCR fallback
# ---------------------------------------------------------------------------

def _ocr_page(page) -> str:
    """Render a PyMuPDF page to image and run pytesseract."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        # 300 DPI is a good trade-off for scientific PDFs
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        print(f"  [OCR] page failed: {e}")
        return ""


def _ocr_document(doc) -> str:
    """Run OCR over every page; used when text extraction yielded near-nothing."""
    if not _OCR_AVAILABLE:
        return ""
    parts = []
    for i, page in enumerate(doc):
        page_text = _ocr_page(page)
        if page_text:
            parts.append(page_text)
        # Safety cap — biomedical reviews rarely need >50 pages OCR'd
        if i >= 50:
            break
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Table extraction (table-aware parsing)
# ---------------------------------------------------------------------------

def _extract_tables(doc, max_pages: int = 30) -> list:
    """
    Extract tables via PyMuPDF find_tables().
    Returns list of dicts: {page, rows: [[cell, ...], ...]}.
    PyMuPDF table support varies — failures are silent (best-effort).
    """
    tables = []
    for page_idx, page in enumerate(doc):
        if page_idx >= max_pages:
            break
        try:
            finder = page.find_tables()
            for tbl in finder.tables:
                rows = tbl.extract()
                if rows and any(any(c for c in r) for r in rows):
                    tables.append({"page": page_idx + 1, "rows": rows})
        except Exception:
            # Older PyMuPDF builds don't have find_tables — ignore.
            continue
    return tables


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extracts text + structured metadata (title, authors, abstract, DOI, year)
    from a PDF. Falls back to OCR when the PDF has no embedded text layer.
    """
    text_content = ""
    page_texts = []        # per-page text (for structure-aware chunking)
    metadata = {}
    title_candidates = []
    used_ocr = False
    tables = []

    try:
        doc = fitz.open(pdf_path)

        # 1. Built-in PDF metadata
        pdf_meta = doc.metadata or {}
        metadata["pdf_title"] = (pdf_meta.get("title") or "").strip()
        metadata["pdf_author"] = (pdf_meta.get("author") or "").strip()
        metadata["pdf_subject"] = (pdf_meta.get("subject") or "").strip()

        # 2. Text per page
        for page in doc:
            ptxt = page.get_text() or ""
            page_texts.append(ptxt)
            text_content += ptxt + "\n"

        # 3. OCR fallback if text layer is missing
        if len(text_content.strip()) < 200 and _OCR_AVAILABLE:
            print(f"  [PDF] Text layer empty — falling back to OCR for {pdf_path}")
            ocr_text = _ocr_document(doc)
            if ocr_text and len(ocr_text) > len(text_content):
                text_content = ocr_text
                page_texts = ocr_text.split("\f") if "\f" in ocr_text else [ocr_text]
                used_ocr = True

        # 4. Title candidates from first page (largest fonts)
        if len(doc) > 0:
            try:
                first_page = doc[0]
                blocks = first_page.get_text(
                    "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
                )["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > 13 and len(span["text"].strip()) > 5:
                                title_candidates.append(
                                    (span["size"], span["text"].strip())
                                )
                title_candidates.sort(key=lambda x: x[0], reverse=True)
            except Exception:
                pass

        # 5. Tables (best-effort, used by structure-aware chunker)
        tables = _extract_tables(doc)

        doc.close()
    except Exception as e:
        return {"error": str(e)}

    # Post-process for OCR artifacts / ligatures / hyphenation
    text_content = _post_process(text_content)
    page_texts = [_post_process(p) for p in page_texts]

    # ---- Heuristic metadata extraction ----
    first_2000 = text_content[:3000]

    title = metadata.get("pdf_title", "")
    if not title and title_candidates:
        title = title_candidates[0][1]
    if not title:
        for line in first_2000.split("\n"):
            line = line.strip()
            if len(line) > 15 and not line.lower().startswith(
                ("http", "doi", "©", "copyright")
            ):
                title = line
                break

    authors = metadata.get("pdf_author", "")
    if not authors:
        author_match = re.search(
            r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+"
            r"(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)*)",
            first_2000,
        )
        if author_match:
            authors = author_match.group(1).strip()

    abstract = ""
    abstract_match = re.search(
        r"(?i)\babstract\b\s*[:\-\n]\s*(.*?)(?=\n\s*\n\s*"
        r"(?:[A-Z][a-z]+\s|1\.\s|Introduction|Background|Keywords|Key\s*words))",
        first_2000, re.DOTALL,
    )
    if abstract_match:
        abstract = re.sub(r"\s+", " ", abstract_match.group(1).strip())

    doi = ""
    doi_match = re.search(r"(10\.\d{4,}/[^\s]+)", text_content[:5000])
    if doi_match:
        doi = doi_match.group(1).rstrip(".,;)")

    year = ""
    year_match = re.search(r"\b((?:19|20)\d{2})\b", first_2000)
    if year_match:
        year = year_match.group(1)

    keywords = []
    kw_match = re.search(
        r"(?i)(?:keywords?|key\s*words?)\s*[:\-]\s*(.+?)(?:\n\n|\n[A-Z])",
        first_2000, re.DOTALL,
    )
    if kw_match:
        keywords = [k.strip() for k in re.split(r"[;,·•]", kw_match.group(1).strip()) if k.strip()]

    return {
        "raw_text": text_content,
        "page_texts": page_texts,   # NEW — per-page for structural chunking
        "tables": tables,           # NEW — table-aware parsing output
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "year": year,
        "keywords": keywords,
        "used_ocr": used_ocr,       # NEW — provenance flag
        "needs_ocr": len(text_content.strip()) < 50,
    }
