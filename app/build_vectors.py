"""CLI script for building the curriculum FAISS index used by the chatbot."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.retrieval import get_embedding_model, get_vectors_path

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _ROOT / "data"
_DEFAULT_MARKDOWN_PATH = _DEFAULT_DATA_DIR / "dsa_requirements.md"
_DEFAULT_JSON_PATH = _DEFAULT_DATA_DIR / "dsa_chs_requirements.json"

_HORIZONTAL_RULE_PATTERN = re.compile(r"^\s*-{3,}\s*$")
_MAX_CHARS_PER_CHUNK = 1200


def _is_table_block(text: str) -> bool:
    """Return True if the block appears to be a markdown table."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return all(line.startswith("|") for line in lines)


def _split_plain_text(text: str, max_chars: int) -> List[str]:
    """Split plain markdown text into chunks below max_chars."""

    chunks: List[str] = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        split_idx = remaining.rfind("\n", 0, max_chars)
        if split_idx == -1:
            split_idx = remaining.rfind(" ", 0, max_chars)
        if split_idx == -1:
            split_idx = max_chars

        chunk = remaining[:split_idx].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _split_section_into_blocks(section: str) -> List[str]:
    """Break a markdown section into semantic blocks."""

    blocks: List[str] = []
    buffer: List[str] = []
    table_mode = False

    def flush_buffer() -> None:
        nonlocal buffer, table_mode
        if buffer:
            block = "\n".join(buffer).strip()
            if block:
                blocks.append(block)
        buffer = []
        table_mode = False

    for line in section.splitlines():
        stripped = line.strip()

        if stripped.startswith("|"):
            if not table_mode:
                flush_buffer()
                table_mode = True
            buffer.append(line)
            continue

        if table_mode and stripped == "":
            buffer.append(line)
            flush_buffer()
            continue

        if stripped == "":
            buffer.append(line)
            flush_buffer()
            continue

        if table_mode:
            flush_buffer()

        buffer.append(line)

    flush_buffer()

    return blocks


def _split_section(section: str, max_chars: int) -> List[str]:
    """Split a section into size-limited chunks while preserving tables."""

    blocks = _split_section_into_blocks(section)
    chunks: List[str] = []
    current_parts: List[str] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current_parts, current_length
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
        current_parts = []
        current_length = 0

    for block in blocks:
        if _is_table_block(block):
            flush_current()
            chunks.append(block)
            continue

        pieces = _split_plain_text(block, max_chars) if len(block) > max_chars else [block]

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate_length = len(piece) if not current_parts else current_length + 2 + len(piece)

            if candidate_length <= max_chars:
                current_parts.append(piece)
                current_length = candidate_length
                continue

            flush_current()

            if len(piece) <= max_chars:
                current_parts.append(piece)
                current_length = len(piece)
            else:
                chunks.append(piece)

    flush_current()

    return [chunk for chunk in chunks if chunk]


def _split_handbook_sections(content: str) -> List[str]:
    """Split handbook content on markdown horizontal rules."""

    sections: List[str] = []
    current: List[str] = []

    for line in content.splitlines():
        if _HORIZONTAL_RULE_PATTERN.match(line):
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = []
        else:
            current.append(line)

    tail = "\n".join(current).strip()
    if tail:
        sections.append(tail)

    return sections


def _load_markdown_documents(handbook_path: Path) -> List[Document]:
    """Load and chunk the handbook markdown while respecting context limits."""

    if not handbook_path.exists():
        raise FileNotFoundError(f"Handbook markdown '{handbook_path}' does not exist.")

    content = handbook_path.read_text(encoding="utf-8")
    sections = _split_handbook_sections(content)

    if not sections:
        raise ValueError(
            "Handbook markdown must contain at least one section separated by a horizontal rule ('---')."
        )

    documents: List[Document] = []

    for section_index, section in enumerate(sections, start=1):
        for chunk_index, chunk in enumerate(
            _split_section(section, _MAX_CHARS_PER_CHUNK), start=1
        ):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": handbook_path.name,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return documents


def _format_modules(modules: Iterable[dict]) -> str:
    return ", ".join(f"{item['code']} ({item['title']})" for item in modules)


def _load_curriculum_documents(json_path: Path) -> List[Document]:
    """Transform the structured CHS curriculum JSON into LangChain documents."""

    if not json_path.exists():
        raise FileNotFoundError(f"Curriculum JSON '{json_path}' does not exist.")

    with json_path.open("r", encoding="utf-8") as handle:
        module_data = json.load(handle)

    chs_curr = module_data["chs_common_curriculum"]
    common_core = chs_curr["common_core"]
    integrated = chs_curr["integrated_courses"]
    interdisciplinary = chs_curr["interdisciplinary_courses"]
    year1_preallocation = chs_curr["year1_preallocation"]

    documents: List[Document] = []

    for pillar in common_core:
        pillar_name = pillar["pillar"]
        if pillar_name == "Communities and Engagement":
            for category in pillar["course_options"]:
                category_name = category["category"]
                subcategories = category["subcategories"]
                for subcategory in subcategories:
                    module_info = subcategory["courses"]
                    modules_text = _format_modules(module_info)

                    if "gen-coded" in subcategory:
                        content = (
                            "Group: CHS Common Core Modules\n"
                            f"Pillar Name: {pillar_name}\n"
                            f"Category: {category_name}\n"
                            f"Gen-Coded: {subcategory['gen-coded']}\n"
                            f"Modules: {modules_text}"
                        )
                    elif "semester" in subcategory:
                        content = (
                            "Group: CHS Common Core Modules\n"
                            f"Pillar Name: {pillar_name}\n"
                            f"Category: {category_name}\n"
                            f"Semester: {subcategory['semester']}\n"
                            f"Modules: {modules_text}"
                        )
                    else:
                        continue

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "group": "CHS Common Core Modules",
                                "pillar": pillar_name,
                                "category": category_name,
                            },
                        )
                    )
        else:
            modules_text = []
            for module in pillar["course_options"]:
                if "footnote" in module:
                    modules_text.append(f"{module['code']} ({module['footnote']})")
                else:
                    modules_text.append(module["code"])

            content = (
                "Group: CHS Common Core Modules\n"
                f"Pillar: {pillar_name}\n"
                f"Modules: {', '.join(modules_text)}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={"group": "CHS Common Core Modules", "pillar": pillar_name},
                )
            )

    for pillar in integrated:
        pillar_name = pillar["pillar"]
        module_codes = ", ".join(mod["code"] for mod in pillar["course_options"])
        content = (
            "Group: CHS Integrated Modules\n"
            f"Pillar: {pillar_name}\n"
            f"Modules: {module_codes}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={"group": "CHS Integrated Modules", "pillar": pillar_name},
            )
        )

    for pillar in interdisciplinary:
        pillar_name = pillar["pillar"]
        module_codes = ", ".join(mod["code"] for mod in pillar["course_options"])
        content = (
            "Group: CHS Interdisciplinary Modules\n"
            f"Pillar: {pillar_name}\n"
            f"Modules: {module_codes}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={"group": "CHS Interdisciplinary Modules", "pillar": pillar_name},
            )
        )

    for student_group, allocation in year1_preallocation.items():
        sem1_modules = ", ".join(allocation["semester_1"])
        sem2_modules = ", ".join(allocation["semester_2"])
        student_label = (
            "Student ID ending with odd number"
            if student_group == "student_id_ending_odd"
            else "Student ID ending with even number"
        )
        content = (
            f"CHS Allocated Student Group: {student_label}\n"
            f"Semester 1 Modules: {sem1_modules}\n"
            f"Semester 2 Modules {sem2_modules}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={"group": "CHS Year 1 Preallocated Modules"},
            )
        )

    return documents


def _build_vector_store(
    documents: List[Document], *, output_dir: Path, overwrite: bool
) -> None:
    """Create and persist the FAISS vector store from documents."""

    embedding = get_embedding_model()

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Vector store directory '{output_dir}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handbook-path",
        type=Path,
        default=_DEFAULT_MARKDOWN_PATH,
        help="Path to the handbook markdown file (default: %(default)s).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=_DEFAULT_JSON_PATH,
        help="Path to the curriculum JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_vectors_path(),
        help="Directory where the FAISS artefacts will be stored (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing vector store directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    handbook_docs = _load_markdown_documents(args.handbook_path)
    curriculum_docs = _load_curriculum_documents(args.json_path)
    documents = handbook_docs + curriculum_docs

    _build_vector_store(documents, output_dir=args.output_dir, overwrite=args.overwrite)

    print(
        "Vector store built successfully with",
        f" {len(handbook_docs)} handbook sections,",
        f" plus {len(curriculum_docs)} structured records."
    )
    print(
        f"Processed handbook markdown '{args.handbook_path}' and saved the index to",
        f" '{args.output_dir}'."
    )

if __name__ == "__main__":
    main()
