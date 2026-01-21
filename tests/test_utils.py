import pytest
from app.utils import parse_questions_json, sha256_bytes

def test_parse_questions_ok():
    raw = b'["  q1  ", "q2", "   "]'
    qs = parse_questions_json(raw)
    assert qs == ["q1", "q2"]

def test_parse_questions_bad_type():
    with pytest.raises(ValueError):
        parse_questions_json(b'{"q":"no"}')

def test_parse_questions_empty():
    with pytest.raises(ValueError):
        parse_questions_json(b'["   "]')

def test_sha256_bytes():
    h = sha256_bytes(b"abc")
    assert len(h) == 64
