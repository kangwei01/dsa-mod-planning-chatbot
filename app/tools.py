from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from .nusmods_client import client

# this gives the information overview of a module
@tool
def nusmods_module_overview(module_code, acad_year = None):
    # provide basic overview information about a module
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

# get prereqs of module
@tool
def nusmods_module_prerequisites(module_code, acad_year = None):
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

# get timetable of module
@tool
def nusmods_module_timetable(module_code, acad_year = None, semester = None, limit_lessons = 20):
    semester_data = client.module_timetable(module_code, acad_year, semester)
    # shape data
    shaped = []
    for sem in semester_data:
        # get lessons for this semester
        lessons = sem.get("timetable", [])
        # if limit specified, truncate lessons
        if limit_lessons is not None:
            lessons = lessons[:limit_lessons]
        # append the semester and lesson times
        shaped.append({
            "semester": sem.get("semester"),
            "lessons": lessons,
        })
    # return the info
    return {
        "moduleCode": client._normalise_code(module_code),
        "acadYear": acad_year or client.default_acad_year,
        "semesterData": shaped,
    }

# search for modules by keyword, level, etc.
@tool
def nusmods_module_search(query, acad_year = None, level = None, limit = 10):
    # use search modules from client
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