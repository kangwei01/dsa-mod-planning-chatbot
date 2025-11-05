from __future__ import annotations
import logging
import requests

# logger --> help with debugging by showing api calls made
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nusmods")

# interact with nusmods api
class NusModsClient:
    def __init__(self, default_acad_year = "2025-2026"):
        self.default_acad_year = default_acad_year # academic year
        self.session = requests.Session() # session so that connections can be reused
        self.module_cache = {} # cache for module details
        self.module_list_cache = {} # cache for module list

    # get base url for that acad year
    def year_base(self, acad_year):
        year = acad_year or self.default_acad_year
        return f"https://api.nusmods.com/v2/{year}"
    
    # normalise the module codes and validate that it is not empty
    def normalise_code(self, module_code):
        # strip any whitespace, convert to uppercase
        code = (module_code or "").strip().upper()
        # if code empty
        if not code:
            # raise error
            raise ValueError("module_code is required")
        # else return the normalized code
        return code
    
    # get module details
    def module(self, module_code, acad_year = None):
        # clean up the module code
        code = self.normalise_code(module_code)
        # get the acad year
        year = acad_year or self.default_acad_year
        # create a cache key --> easier retrieval
        cache_key = f"{year}:{code}"
        # if not in cache
        if cache_key not in self._module_cache:
            # construct the url
            url = f"{self.year_base(year)}/modules/{code}.json"
            # log the api call
            logger.info("Fetching module details: %s", url)
            # make the api call
            response = self.session.get(url)
            # raise error if bad response
            response.raise_for_status()
            # store the result in cache
            self.module_cache[cache_key] = response.json()
        # return the cached module details
        return self.module_cache[cache_key]

    # get list of modules for academic year
    def module_list(self, acad_year = None):
        year = acad_year or self.default_acad_year
        # if year not in the cache for module list
        if year not in self.module_list_cache:
            # construct the url
            url = f"{self.year_base(year)}/moduleList.json"
            # log the api call
            logger.info("Fetching module list: %s", url)
            # make the api call
            response = self.session.get(url)
            # raise error if bad response
            response.raise_for_status()
            # store the result in cache
            self.module_list_cache[year] = response.json()
        # return the cached module list --> if aleady cached, then no api call made --> faster
        return self.module_list_cache[year]

    # create a search function to find modules by code or title
    def search_modules(self, query, acad_year = None, level = None, limit = 10):
        # make query lowercase and strip whitespace
        query_lower = (query or "").strip().lower()
        # if string empty raise error
        if not query_lower:
            raise ValueError("query must be a non-empty string")
        
        # store all matches
        matches = []
        # for mods in this AY
        for mod in self.module_list(acad_year):
            # if level specified, filter by level
            # level: 1, 2, 3, 4, etc. (e.g.: MA2002 -> level 2, DSA4213 -> level 4)
            # this is just in case caller explicitly asks for a level (e.g.: level 1000 mod)
            if level is not None:
                # get the module code
                code = mod.get("moduleCode", "")
                # check if code valid: at least 3 characters, 3rd is digit
                if len(code) >= 3 and code[2].isdigit():
                    # check if level matches
                    if int(code[2]) != int(level):
                        # skip this module
                        continue
                else:
                    continue
            # check if query matches module code or title
            if query_lower in mod.get("moduleCode", "").lower() or query_lower in mod.get("title", "").lower():
                matches.append(mod)
            # if reached limit, stop searching
            if len(matches) >= limit:
                break
        return matches
    
    # get timetables
    def module_timetable(self, module_code, acad_year = None, semester = None):
        # get module code details
        data = self.module(module_code, acad_year)
        # get semester data
        semester_data = data.get("semesterData", [])
        # if no semester specified, return all
        if semester is None:
            return semester_data
        # else filter by semester
        return [sem for sem in semester_data if sem.get("semester") == semester]

client = NusModsClient()