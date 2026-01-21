import hashlib
import json
from fastapi import UploadFile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def read_upload_bytes(file: UploadFile) -> bytes:
    data = await file.read()
    await file.seek(0)
    return data


def parse_questions_json(raw: bytes) -> list[str]:
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError("questions.json must be a JSON array of strings")

    cleaned = [q.strip() for q in obj if q.strip()]
    if not cleaned:
        raise ValueError("questions.json must contain at least one non-empty question")

    return cleaned
