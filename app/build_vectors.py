"""CLI script for building the curriculum FAISS index used by the chatbot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retrieval import get_embedding_model, get_vectors_path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PDF_DIR = _ROOT / "data"
_DEFAULT_JSON_PATH = _DEFAULT_PDF_DIR / "dsa_chs_requirements.json"


def _load_pdf_documents(pdf_dir: Path) -> tuple[List[Document], int]:
    """Load all PDF documents within *pdf_dir* using LangChain loaders."""

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' does not exist.")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")
    documents: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for doc in pages:
            doc.metadata.setdefault("source", path.name)
        documents.extend(pages)

    return documents, len(pdf_paths)


def _split_documents(
    documents: Iterable[Document], *, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split documents into smaller chunks compatible with embedding limits."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(list(documents))


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
        "--pdf-dir",
        type=Path,
        default=_DEFAULT_PDF_DIR,
        help="Directory containing source PDF handbooks (default: %(default)s).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=_DEFAULT_JSON_PATH,
        help="Path to the curriculum JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters when splitting PDF content.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters when splitting PDF content.",
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

    pdf_docs, pdf_file_count = _load_pdf_documents(args.pdf_dir)
    pdf_chunks = _split_documents(
        pdf_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    curriculum_docs = _load_curriculum_documents(args.json_path)
    documents = pdf_chunks + curriculum_docs

    _build_vector_store(documents, output_dir=args.output_dir, overwrite=args.overwrite)

    print(
        "Vector store built successfully with"
        f" {len(pdf_docs)} PDF pages split into {len(pdf_chunks)} chunks,"
        f" plus {len(curriculum_docs)} structured records."
    )
    print(
        f"Processed {pdf_file_count} PDF files in '{args.pdf_dir}' and saved the index to"
        f" '{args.output_dir}'."
    )


if __name__ == "__main__":
    main()
