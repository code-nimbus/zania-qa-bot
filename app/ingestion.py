import io
import json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_bytes(pdf_bytes: bytes) -> List[Document]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    docs: List[Document] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": "pdf"},
                )
            )
    return docs


def load_json_bytes(json_bytes: bytes) -> List[Document]:
    obj = json.loads(json_bytes.decode("utf-8"))
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return [Document(page_content=text, metadata={"source": "json"})]


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
