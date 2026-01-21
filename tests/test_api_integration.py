import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.rag as rag
import app.main as main_mod
from app.main import app

client = TestClient(app)


class FakeLLM:
    def invoke(self, *args, **kwargs):
        class R:
            content = "FAKE_ANSWER"
        return R()

    def predict(self, *args, **kwargs):
        return "FAKE_ANSWER"

    def __call__(self, *args, **kwargs):
        return "FAKE_ANSWER"


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


def patch_answerer(monkeypatch):
    """
    Patch the actual function used by the /v1/qa route.
    Depending on how app.main imports it, we need to patch:
      - app.rag.<fn>
      - app.main.<fn>
    We patch a few common names to cover your implementation.
    """
    def _fake_answer(*args, **kwargs):
        return "FAKE_ANSWER"

    candidate_names = [
        "answer_question",
        "answer_questions",
        "run_qa",
        "qa",
        "qa_over_docs",
        "run_rag",
    ]

    for name in candidate_names:
        if hasattr(rag, name):
            monkeypatch.setattr(rag, name, _fake_answer, raising=False)
        if hasattr(main_mod, name):
            monkeypatch.setattr(main_mod, name, _fake_answer, raising=False)


def test_v1_qa_json_doc_offline(monkeypatch, tmp_path):
    from app import settings as settings_mod
    settings_mod.settings.CHROMA_PERSIST_DIR = str(tmp_path)

    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(rag, "get_embeddings", lambda: FakeEmbeddings())

    # ✅ Patch the actual answer function used by the endpoint (rag OR main)
    patch_answerer(monkeypatch)

    root = Path(__file__).resolve().parents[1]
    sample_json = root / "sample.json"
    assert sample_json.exists(), "sample.json not found in project root"

    questions = io.BytesIO(json.dumps(["What is a?", "What is b?"]).encode("utf-8"))
    doc = io.BytesIO(sample_json.read_bytes())

    r = client.post(
        "/v1/qa",
        files={
            "questions_file": ("questions.json", questions, "application/json"),
            "document_file": ("sample.json", doc, "application/json"),
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) >= 1
    assert body["results"][0]["answer"] == "FAKE_ANSWER"


def test_v1_qa_pdf_doc_offline(monkeypatch, tmp_path):
    from app import settings as settings_mod
    settings_mod.settings.CHROMA_PERSIST_DIR = str(tmp_path)

    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(rag, "get_embeddings", lambda: FakeEmbeddings())

    # ✅ Patch the actual answer function used by the endpoint (rag OR main)
    patch_answerer(monkeypatch)

    root = Path(__file__).resolve().parents[1]
    soc2_pdf = root / "soc2-type2.pdf"
    assert soc2_pdf.exists(), "soc2-type2.pdf not found in project root"

    questions = io.BytesIO(json.dumps(["What is this report about?"]).encode("utf-8"))
    doc = io.BytesIO(soc2_pdf.read_bytes())

    r = client.post(
        "/v1/qa",
        files={
            "questions_file": ("questions.json", questions, "application/json"),
            "document_file": ("soc2-type2.pdf", doc, "application/pdf"),
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) >= 1
    assert body["results"][0]["answer"] == "FAKE_ANSWER"


def test_v1_qa_rejects_wrong_doc_type():
    questions = io.BytesIO(b'["q1"]')
    doc = io.BytesIO(b"nope")

    r = client.post(
        "/v1/qa",
        files={
            "questions_file": ("questions.json", questions, "application/json"),
            "document_file": ("doc.txt", doc, "text/plain"),
        },
    )
    assert r.status_code == 400
