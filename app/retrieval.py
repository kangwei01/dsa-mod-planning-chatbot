"""Retrieval helpers that mirror the notebook prototype implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from ollama import Client
from ollama._types import ResponseError

# Vector store artefacts live within the ``app`` package so deployments can
# treat the package as a self-contained module without relying on repository
# relative paths.
_VECTORS_PATH = Path(__file__).resolve().parent / "curriculum_info_vectors"
_EMBED_MODEL = "mxbai-embed-large"


@lru_cache(maxsize=1)
def _ensure_embedding_model(model: str) -> None:
    """Confirm the Ollama embedding model exists, pulling it when absent."""

    client = Client()
    try:
        client.show(model=model)
    except ResponseError as exc:
        message = str(exc).lower()
        if "not found" in message or "404" in message:
            try:
                client.pull(model=model, stream=False)
            except ResponseError as pull_exc:
                raise RuntimeError(
                    f"Embedding model '{model}' is missing and could not be pulled automatically. "
                    f"Run `ollama pull {model}` on the host to install it before enabling retrieval."
                ) from pull_exc
            try:
                client.show(model=model)
            except ResponseError as verify_exc:
                raise RuntimeError(
                    f"Embedding model '{model}' could not be verified after pulling."
                ) from verify_exc
        else:
            raise RuntimeError(
                f"Failed to verify Ollama embedding model '{model}'."
            ) from exc
    except ConnectionError as exc:
        raise RuntimeError(
            "Could not reach the local Ollama service while checking for the embedding model. "
            "Ensure Ollama is running and accessible."
        ) from exc


@lru_cache(maxsize=1)
def _embedding() -> OllamaEmbeddings:
    """Return a singleton Ollama embedding model."""

    _ensure_embedding_model(_EMBED_MODEL)
    return OllamaEmbeddings(model=_EMBED_MODEL)


def get_embedding_model() -> OllamaEmbeddings:
    """Expose the embedding model factory for tooling outside this module."""

    return _embedding()


def get_vectors_path() -> Path:
    """Return the path where FAISS artefacts are stored."""

    return _VECTORS_PATH


@lru_cache(maxsize=1)
def _vectorstore() -> FAISS:
    """Load the FAISS vector store built during development notebooks."""

    if not _VECTORS_PATH.exists():
        raise FileNotFoundError(
            "curriculum_info_vectors directory is missing. Run `python -m "
            "app.build_vectors` to generate the FAISS index before enabling "
            "retrieval."
        )

    embedding = _embedding()
    return FAISS.load_local(
        str(_VECTORS_PATH),
        embedding,
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """Expose the FAISS retriever used by the LangGraph nodes."""

    vectorstore = _vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


def format_documents(documents: Sequence[Document] | None, *, max_chars: int = 500) -> str:
    """Convert retrieved documents into a compact newline-separated context block."""

    if not documents:
        return ""

    snippets: List[str] = []
    for index, doc in enumerate(documents, start=1):
        text = doc.page_content.strip()
        if max_chars is not None and max_chars > 0:
            text = text[: max_chars].rstrip()
        snippets.append(f"Doc {index}: {text}")
    return "\n\n".join(snippets)


def combine_context(documents: Sequence[Document] | None) -> str:
    """Combine document contents into a single context string for augmentation."""

    if not documents:
        return ""

    parts: Iterable[str] = (doc.page_content.strip() for doc in documents)
    return "\n\n".join(part for part in parts if part)


__all__ = [
    "combine_context",
    "format_documents",
    "get_embedding_model",
    "get_retriever",
    "get_vectors_path",
]
