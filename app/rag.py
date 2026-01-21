from __future__ import annotations

from typing import List, Protocol
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

from .settings import settings
from .utils import sha256_text


SYSTEM = """You are a QA bot. Answer ONLY using the provided context.
If the context does not contain the answer, say exactly: "Not found in document."
Be concise and specific. If relevant, cite page numbers from metadata.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class EmbeddingsLike(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class LLMLike(Protocol):
    def invoke(self, messages): ...


def get_embeddings() -> EmbeddingsLike:
    from langchain_openai import OpenAIEmbeddings
    # Will pick up OPENAI_API_KEY from env (loaded by settings.py)
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_llm() -> LLMLike:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
    )


def get_vectorstore(collection_name: str, embeddings: EmbeddingsLike) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def _chunk_ids(chunks: List[Document]) -> List[str]:
    ids: List[str] = []
    for c in chunks:
        meta = str(sorted(c.metadata.items()))
        ids.append(sha256_text(c.page_content + meta)[:32])
    return ids


def upsert_documents(collection_name: str, chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    vs = get_vectorstore(collection_name, embeddings)

    lock = FileLock(f"{settings.CHROMA_PERSIST_DIR}/{collection_name}.lock")
    with lock:
        ids = _chunk_ids(chunks)
        current = vs._collection.count()
        if current == 0:
            vs.add_documents(chunks, ids=ids)
            vs.persist()

    return vs


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
)
def answer_question(vs: Chroma, question: str) -> str:
    llm = get_llm()
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K})

    # ✅ LangChain compatibility: some versions use .invoke(), some use get_relevant_documents()
    if hasattr(retriever, "invoke"):
        retrieved = retriever.invoke(question)
    else:
        retrieved = retriever.get_relevant_documents(question)

    context = "\n\n".join(f"[meta={d.metadata}] {d.page_content}" for d in retrieved)

    messages = PROMPT.format_messages(context=context, question=question)
    resp = llm.invoke(messages)
    return str(resp.content).strip()
