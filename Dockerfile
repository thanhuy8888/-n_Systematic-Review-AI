# ─── Stage 1: Build ────────────────────────────────────────────────────────
# Python 3.10 slim — keeps image size reasonable
FROM python:3.10-slim AS base

WORKDIR /app

# System deps for PyMuPDF + torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libmupdf-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache — rebuilds only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Pre-download ML models ──────────────────────────────────────
# Models are cached to /root/.cache/huggingface/hub
# They persist via a named Docker volume (see docker-compose.yml)
# This RUN step runs only on first build; subsequent builds use cache.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                         AutoModelForQuestionAnswering; \
print('[MODEL] Downloading SciBERT...'); \
AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'); \
AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=2); \
print('[MODEL] Downloading RoBERTa-SQuAD2...'); \
AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); \
AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2'); \
print('[MODEL] All models ready.')"

# ─── Stage 3: App ─────────────────────────────────────────────────────────
COPY . .

# SQLite database directory
RUN mkdir -p /app/data/raw /app/data/processed /app/data/uploads

EXPOSE 8000

# Healthcheck so docker-compose frontend waits for backend
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
