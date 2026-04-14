# AI-Powered Systematic Review Platform

This platform is a comprehensive AI-assisted tool designed to accelerate the Systematic Literature Review (SLR) process. It leverages an advanced local HuggingFace Transformer architecture to automate the screening of academic papers (abstract and full-text) based on custom PICO criteria, and extracts structured data using RoBERTa Extractive Question Answering.

## 🚀 Features
- **PDF Metadata Ingestion**: Automatically extracts metadata (Title, Authors, Abstract, DOI) from uploaded PDF articles.
- **Criteria-Aware Screening**: Uses AI to screen abstracts and full-texts against your specific Research Protocol (PICO).
- **Explainable AI (XAI)**: Provides transparent rationale and confidence scores for inclusion/exclusion decisions.
- **Data Extraction**: Automatically answers methodology, sample size, limitations, and risk of bias questions using Extractive QA on the paper content.
- **Local Privacy**: Runs `SciBERT` and `RoBERTa-SQuAD2` models completely locally, ensuring data privacy and no reliance on paid API services like OpenAI.

## 🗂️ Project Structure
```text
/
├── apps/
│   ├── api/        # FastAPI Python Backend (Model inference, database)
│   └── web/        # React + TypeScript Frontend (Vite)
├── sr_core/        # Native Python core logic (PDF parsing, Transformer, QA engine)
├── requirements.txt # Python backend dependencies
└── README.md       # This file
```

---

## 🛠️ Installation & Setup

Before running the project, you must install the dependencies for both the Python Backend and the React Frontend.

### 1. Prerequisites
- **Python**: Version `3.10` or newer is required.
- **Node.js**: Version `18.0` or newer is required.
- **Git**

### 2. Backend Setup (Python)
Open a terminal in the root folder (where this file is located):

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate

# 3. Install the required Python packages
pip install -r requirements.txt
```
*(Note: The first time you run the backend, it will automatically download the HuggingFace models `allenai/scibert_scivocab_uncased` and `deepset/roberta-base-squad2`. This requires internet access and ~1GB of disk space).*

### 3. Frontend Setup (React/Node.js)
Open a **new** terminal, navigate to the `apps/web` directory:

```bash
# 1. Navigate to the frontend directory
cd apps/web

# 2. Install Node packages
npm install
```

---

## 🏃‍♂️ Running the Application

To use the software, you need to run **both** the backend and the frontend simultaneously in two separate terminals.

### Step 1: Start the Backend (FastAPI + AI Models)
In your first terminal (ensure your `venv` is activated):
```bash
# From the root directory:
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`.*

### Step 2: Start the Frontend (React UI)
In your second terminal (inside the `apps/web` directory):
```bash
# From the apps/web directory:
npm run dev
```
*The Web UI will be available at `http://localhost:5173`.*

---

## 📖 Usage Guide
1. Open the UI at `http://localhost:5173`.
2. **Review Protocol**: Enter your PICO (Population, Intervention, Comparison, Outcome) criteria.
3. **Identification**: Upload academic PDF files. The system will automatically parse the metadata.
4. **Abstract Screening**: Click "Run Abstract Screening" to let the core Transformer model filter out highly irrelevant papers based on your criteria.
5. **Full-text Screening**: For included papers, click "Run Full-text Screening" for a deeper semantic analysis using the full PDF content.
6. **Extraction**: Click "Run Extraction" to generate an evidence table containing Methodology, Sample Size, Key Findings, etc.

## 🤝 Contributing
Contributions are welcome. Please ensure all modifications pass the standard TypeScript linters and Python `pytest` suites before submitting a pull request.
