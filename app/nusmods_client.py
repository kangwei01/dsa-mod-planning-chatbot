"""Utility helpers for interacting with the public NUSMods API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

# Configure a dedicated logger so HTTP calls can be inspected easily when debugging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nusmods")


class NusModsClient:
    """Thin wrapper around the NUSMods v2 API with basic response caching."""

    def __init__(self, default_acad_year: str = "2025-2026") -> None:
        # Default academic year so tools can omit it unless a user provides one.
        self.default_acad_year = default_acad_year
        # Shared ``Session`` instance keeps TCP connections warm for faster calls.
        self._session = requests.Session()
        # Lightweight in-memory caches avoid repeated work within a single process.
        self._module_cache: Dict[str, Dict[str, Any]] = {}
        self._module_list_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _year_base(self, acad_year: Optional[str]) -> str:
        """Return the base URL for the selected academic year."""

        year = acad_year or self.default_acad_year
        return f"https://api.nusmods.com/v2/{year}"

    def _normalise_code(self, module_code: str) -> str:
        """Normalise and validate module codes received from the caller."""

        code = (module_code or "").strip().upper()
        if not code:
            raise ValueError("module_code is required")
        return code

    def module(self, module_code: str, acad_year: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve the canonical module payload for a course."""

        code = self._normalise_code(module_code)
        year = acad_year or self.default_acad_year
        cache_key = f"{year}:{code}"
        if cache_key not in self._module_cache:
            url = f"{self._year_base(year)}/modules/{code}.json"
            logger.info("Fetching module details: %s", url)
            response = self._session.get(url)
            response.raise_for_status()
            self._module_cache[cache_key] = response.json()
        return self._module_cache[cache_key]

    def module_list(self, acad_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the entire module catalogue for keyword searches."""

        year = acad_year or self.default_acad_year
        if year not in self._module_list_cache:
            url = f"{self._year_base(year)}/moduleList.json"
            logger.info("Fetching module list: %s", url)
            response = self._session.get(url)
            response.raise_for_status()
            self._module_list_cache[year] = response.json()
        return self._module_list_cache[year]

    def search_modules(
        self,
        query: str,
        acad_year: Optional[str] = None,
        level: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for modules using a keyword and optional level filter."""

        query_lower = (query or "").strip().lower()
        if not query_lower:
            raise ValueError("query must be a non-empty string")

        matches: List[Dict[str, Any]] = []
        for mod in self.module_list(acad_year):
            if level is not None:
                code = mod.get("moduleCode", "")
                if len(code) >= 3 and code[2].isdigit():
                    if int(code[2]) != int(level):
                        continue
                else:
                    continue
            if query_lower in mod.get("moduleCode", "").lower() or query_lower in mod.get("title", "").lower():
                matches.append(mod)
            if len(matches) >= limit:
                break
        return matches

    def module_timetable(
        self,
        module_code: str,
        acad_year: Optional[str] = None,
        semester: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the timetable blocks for a module."""

        data = self.module(module_code, acad_year)
        semester_data = data.get("semesterData", [])
        if semester is None:
            return semester_data
        return [sem for sem in semester_data if sem.get("semester") == semester]


# Shared client instance reused across the tool definitions.
client = NusModsClient()

__all__ = ["NusModsClient", "client"]
