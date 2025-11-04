"""LangChain tool definitions used by the chatbot graph."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from langchain_core.tools import tool

from .nusmods_client import client


DATA_DIR = Path(__file__).resolve().parent
_EMBED_DIMENSION = 512
_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    """Normalise vectors to unit length for cosine similarity."""

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _tokenise(text: str) -> Iterable[str]:
    """Yield lowercase tokens from ``text`` using a simple word boundary split."""

    return (token.lower() for token in _TOKEN_PATTERN.findall(text))


def _hash_token(token: str) -> int:
    """Map a token to a stable bucket using SHA1 hashing."""

    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % _EMBED_DIMENSION


def _embed_text(text: str) -> np.ndarray:
    """Create a deterministic sparse embedding suitable for FAISS indexing."""

    vector = np.zeros(_EMBED_DIMENSION, dtype=np.float32)
    for token in _tokenise(text):
        bucket = _hash_token(token)
        vector[bucket] += 1.0
    return _normalise_vector(vector)


def _embed_batch(texts: Sequence[str]) -> np.ndarray:
    """Embed a list of texts into a 2D array understood by FAISS."""

    if not texts:
        return np.empty((0, _EMBED_DIMENSION), dtype=np.float32)
    matrix = np.vstack([_embed_text(text) for text in texts]).astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix


def _is_scalar(value: Any) -> bool:
    """Return ``True`` if ``value`` is a scalar that can form a chunk leaf."""

    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
        return True
    return False


def _format_scalar(value: Any) -> str:
    """Convert scalar and near-scalar values into a readable string."""

    if isinstance(value, list):
        return ", ".join("" if item is None else str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _load_json_chunks(data: Any, path: Tuple[str, ...]) -> List[Tuple[str, Dict[str, Any]]]:
    """Recursively flatten JSON data into path-aware text chunks."""

    chunks: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(data, dict):
        if data and all(_is_scalar(value) for value in data.values()):
            lines = [" > ".join(path)]
            lines.extend(f"{key}: {_format_scalar(value)}" for key, value in data.items())
            chunks.append(("\n".join(lines), {"path": " > ".join(path)}))
        else:
            for key, value in data.items():
                chunks.extend(_load_json_chunks(value, path + (str(key),)))
    elif isinstance(data, list):
        if data and all(_is_scalar(item) for item in data):
            text = f"{' > '.join(path)}: {_format_scalar(data)}"
            chunks.append((text, {"path": " > ".join(path)}))
        else:
            for index, item in enumerate(data):
                label = str(index)
                if isinstance(item, dict):
                    if "code" in item:
                        label = str(item["code"])
                    elif "title" in item:
                        label = str(item["title"])
                    elif "pillar" in item:
                        label = str(item["pillar"])
                chunks.extend(_load_json_chunks(item, path + (label,)))
    else:
        chunks.append((f"{' > '.join(path)}: {_format_scalar(data)}", {"path": " > ".join(path)}))
    return chunks


def _load_chs_requirements() -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load and chunk the CHS requirements JSON document."""

    with (DATA_DIR / "dsa_chs_requirements.json").open("r", encoding="utf-8") as file:
        raw = json.load(file)

    base_path: Tuple[str, ...] = ("College of Humanities and Sciences Requirements",)
    chunks = _load_json_chunks(raw, base_path)
    texts, metadatas = zip(*chunks) if chunks else ([], [])
    return list(texts), list(metadatas)


def _load_major_requirements() -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load and chunk the DSA major requirements Markdown document."""

    content = (DATA_DIR / "dsa_major_requirements.md").read_text(encoding="utf-8")
    parts = [segment.strip() for segment in re.split(r"\n\s*---\s*\n", content) if segment.strip()]

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for index, segment in enumerate(parts):
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0]
        texts.append("\n".join(lines))
        metadatas.append({"section": title, "index": index})
    return texts, metadatas


@dataclass
class _VectorStore:
    """Lightweight FAISS-backed similarity search helper."""

    texts: Sequence[str]
    metadatas: Sequence[Dict[str, Any]]
    index: faiss.Index

    @classmethod
    def from_texts(cls, texts: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> "_VectorStore":
        vectors = _embed_batch(texts)
        index = faiss.IndexFlatIP(_EMBED_DIMENSION)
        if len(vectors):
            index.add(vectors)
        return cls(texts=texts, metadatas=metadatas, index=index)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self.texts:
            return []
        vector = _embed_batch([query])
        if vector.size == 0:
            return []
        scores, indices = self.index.search(vector, min(top_k, len(self.texts)))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            metadata = dict(self.metadatas[idx])
            metadata["score"] = float(score)
            results.append({
                "content": self.texts[idx],
                "metadata": metadata,
            })
        return results


_CHS_TEXTS, _CHS_METADATA = _load_chs_requirements()
_CHS_STORE = _VectorStore.from_texts(_CHS_TEXTS, _CHS_METADATA)
_CHS_TOP_K = 4
_MAJOR_TEXTS, _MAJOR_METADATA = _load_major_requirements()
_MAJOR_STORE = _VectorStore.from_texts(_MAJOR_TEXTS, _MAJOR_METADATA)
_MAJOR_TOP_K = 1


@tool
def chs_requirements_lookup(query: str) -> Dict[str, Any]:
    """Retrieve College of Humanities and Sciences requirements context before using module APIs."""

    results = _CHS_STORE.search(query, top_k=_CHS_TOP_K)
    return {"query": query, "matches": results}


@tool
def dsa_major_requirements_lookup(query: str) -> Dict[str, Any]:
    """Retrieve Data Science and Analytics major requirements context prior to calling module APIs."""

    results = _MAJOR_STORE.search(query, top_k=_MAJOR_TOP_K)
    return {"query": query, "matches": results}


@tool
def nusmods_module_overview(module_code: str, acad_year: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve the canonical module payload for course planning questions.

    `acad_year`, when provided, should follow the YYYY-YYYY format (for example,
    `2024-2025`)."""

    data = client.module(module_code, acad_year)
    return {
        "moduleCode": data.get("moduleCode"),
        "title": data.get("title"),
        "description": data.get("description"),
        "moduleCredit": data.get("moduleCredit"),
        "faculty": data.get("faculty"),
        "department": data.get("department"),
        "prerequisite": data.get("prerequisite"),
        "preclusion": data.get("preclusion"),
        "fulfillRequirements": data.get("fulfillRequirements"),
    }


@tool
def nusmods_module_prerequisites(module_code: str, acad_year: Optional[str] = None) -> Dict[str, Any]:
    """Surface prerequisite, preclusion, and fulfilment data for a module.

    `acad_year`, when provided, should follow the YYYY-YYYY format (for example,
    `2024-2025`)."""

    data = client.module(module_code, acad_year)
    return {
        "moduleCode": data.get("moduleCode"),
        "title": data.get("title"),
        "prerequisite": data.get("prerequisite"),
        "prerequisiteTree": data.get("prerequisiteTree"),
        "fulfillRequirements": data.get("fulfillRequirements"),
        "preclusion": data.get("preclusion"),
        "corequisite": data.get("corequisite"),
    }


@tool
def nusmods_module_timetable(
    module_code: str,
    acad_year: Optional[str] = None,
    semester: Optional[int] = None,
    limit_lessons: Optional[int] = 20,
) -> Dict[str, Any]:
    """Summarise the module timetable across semesters and lesson groupings.

    `acad_year`, when provided, should follow the YYYY-YYYY format (for example,
    `2024-2025`) and `semester`, when provided, should be the integer `1` or `2`."""

    semester_data = client.module_timetable(module_code, acad_year, semester)
    shaped: List[Dict[str, Any]] = []
    for sem in semester_data:
        lessons = sem.get("timetable", [])
        if limit_lessons is not None:
            lessons = lessons[:limit_lessons]
        shaped.append({
            "semester": sem.get("semester"),
            "lessons": lessons,
        })
    return {
        "moduleCode": client._normalise_code(module_code),
        "acadYear": acad_year or client.default_acad_year,
        "semesterData": shaped,
    }


@tool
def nusmods_module_search(
    query: str,
    acad_year: Optional[str] = None,
    level: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Locate modules by keyword, optionally filtered by level, for discovery tasks.

    `acad_year`, when provided, should follow the YYYY-YYYY format (for example,
    `2024-2025`)."""

    matches = client.search_modules(query, acad_year, level=level, limit=limit)
    return {
        "query": query,
        "acadYear": acad_year or client.default_acad_year,
        "count": len(matches),
        "results": [
            {
                "moduleCode": mod.get("moduleCode"),
                "title": mod.get("title"),
                "moduleCredit": mod.get("moduleCredit"),
            }
            for mod in matches
        ],
    }


API_TOOLS = [
    nusmods_module_overview,
    nusmods_module_prerequisites,
    nusmods_module_timetable,
    nusmods_module_search,
]

RETRIEVAL_TOOLS = [
    chs_requirements_lookup,
    dsa_major_requirements_lookup,
]

ALL_TOOLS = RETRIEVAL_TOOLS + API_TOOLS

__all__ = [
    "API_TOOLS",
    "ALL_TOOLS",
    "RETRIEVAL_TOOLS",
    "chs_requirements_lookup",
    "dsa_major_requirements_lookup",
    "nusmods_module_overview",
    "nusmods_module_prerequisites",
    "nusmods_module_search",
    "nusmods_module_timetable",
]
