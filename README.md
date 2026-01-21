# Zania QA Bot (FastAPI + LangChain + Chroma)

Backend API that answers a list of questions based on an input document (PDF or JSON) using RAG.

## What it does
- Upload `questions.json` (JSON array of strings)
- Upload a document (`.pdf` or `.json`)
- Returns answers in JSON

## Setup
```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
