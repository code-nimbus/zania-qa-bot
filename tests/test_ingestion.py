from pathlib import Path
from app.ingestion import load_json_bytes, chunk_docs


def test_load_json_bytes_object():
    docs = load_json_bytes(b'{"a": 1, "b": {"c": 2}}')
    assert len(docs) == 1
    assert "a" in docs[0].page_content


def test_chunk_docs_nonempty():
    # Use only real file: sample.json (project root)
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "sample.json"
    assert sample_path.exists(), "sample.json not found in project root"

    docs = load_json_bytes(sample_path.read_bytes())
    chunks = chunk_docs(docs)

    assert len(chunks) >= 1
    assert all(c.page_content.strip() for c in chunks)
