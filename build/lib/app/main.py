import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request

from .schemas import QAResponse, QAItem
from .settings import settings
from .utils import read_upload_bytes, parse_questions_json, sha256_bytes
from .ingestion import load_pdf_bytes, load_json_bytes, chunk_docs
from .rag import upsert_documents, answer_question
from .logging_mw import RequestIdMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zania")

app = FastAPI(title="Zania QA Bot", version="1.0.0")
app.add_middleware(RequestIdMiddleware)


@app.get("/health")
def health():
    return {"status": "ok"}


def _check_size(name: str, data: bytes) -> None:
    if len(data) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{name} exceeds max size {settings.MAX_FILE_SIZE_BYTES} bytes",
        )


@app.post("/v1/qa", response_model=QAResponse)
async def qa(
    request: Request,
    questions_file: UploadFile = File(..., description="JSON array of questions"),
    document_file: UploadFile = File(..., description="PDF or JSON document"),
):
    q_bytes = await read_upload_bytes(questions_file)
    d_bytes = await read_upload_bytes(document_file)

    _check_size("questions_file", q_bytes)
    _check_size("document_file", d_bytes)

    # Validate questions
    try:
        questions = parse_questions_json(q_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(questions) > settings.MAX_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many questions: {len(questions)} (max {settings.MAX_QUESTIONS})",
        )

    # Validate document type by filename
    doc_name = (document_file.filename or "").lower()
    if doc_name.endswith(".pdf"):
        docs = load_pdf_bytes(d_bytes)
    elif doc_name.endswith(".json"):
        try:
            docs = load_json_bytes(d_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON document: {e}")
    else:
        raise HTTPException(status_code=400, detail="document_file must be .pdf or .json")

    if not docs:
        raise HTTPException(status_code=400, detail="No extractable text found in document")

    chunks = chunk_docs(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks")

    doc_hash = sha256_bytes(d_bytes)[:16]
    collection = f"doc_{doc_hash}"

    logger.info(
        "ingest start",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "collection": collection,
            "questions": len(questions),
            "chunks": len(chunks),
        },
    )

    vs = upsert_documents(collection, chunks)

    results = []
    for q in questions:
        try:
            ans = answer_question(vs, q)
        except Exception as e:
            ans = f"Error answering question: {type(e).__name__}"
        results.append(QAItem(question=q, answer=ans))

    return QAResponse(results=results)
