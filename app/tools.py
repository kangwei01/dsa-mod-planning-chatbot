"""LangChain tool definitions used by the chatbot graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from .nusmods_client import client


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

__all__ = [
    "API_TOOLS",
    "nusmods_module_overview",
    "nusmods_module_prerequisites",
    "nusmods_module_search",
    "nusmods_module_timetable",
]
