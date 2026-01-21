from pydantic import BaseModel, Field
from typing import List


class QAItem(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    results: List[QAItem]


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Invalid file type"])
