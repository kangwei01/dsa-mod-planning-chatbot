from langchain_core.documents import Document
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from retrieval import embedding


# We split the CHS Requirements first from json into Langchain documents
chs_path = "data/dsa_chs_requirements.json"

with open(chs_path, "r", encoding="utf-8") as f:
    module_data = json.load(f)

json_docs = []

chs_curr = module_data['chs_common_curriculum']

common_core = chs_curr['common_core']
integrated = chs_curr['integrated_courses']
interdisciplinary = chs_curr['interdisciplinary_courses']
year1_preallocation = chs_curr['year1_preallocation']

chs_curr_docs = []

for pillars in common_core:
    pillar_name = pillars['pillar']
    if pillar_name == 'Communities and Engagement':
        course_options = pillars['course_options']

        for categories in course_options:
            category_name = categories['category']
            subcategories = categories['subcategories']

            for sub in subcategories:
                if 'gen-coded' in sub:
                    gen_coded = sub.get('gen-coded')
                    module_info = sub['courses']

                    modules_text = []
                    
                    for mod in module_info:
                        modules_text.append(f"{mod['code']} ({mod['title']})")
                    
                    content = (
                        f"Group: CHS Common Core Modules\n"
                        f"Pillar Name: {pillar_name}\n"
                        f"Category: {category_name}\n"
                        f"Gen-Coded: {gen_coded}\n"
                        f"Modules: {', '.join(modules_text)}"
                    )
                
                elif 'semester' in sub:
                    sem = sub.get('semester')
                    module_info = sub['courses']

                    modules_text = []
                    
                    for mod in module_info:
                        modules_text.append(f"{mod['code']} ({mod['title']})")
                    
                    content = (
                        f"Group: CHS Common Core Modules\n"
                        f"Pillar Name: {pillar_name}\n"
                        f"Category: {category_name}\n"
                        f"Semester: {sem}\n"
                        f"Modules: {', '.join(modules_text)}"
                    )

                chs_curr_docs.append(
                    Document(
                        page_content=content,
                        metadata = {"group": "CHS Common Core Modules", "pillar": pillar_name, "category": category_name}
                    )
                )
    else:
        course_options = pillars['course_options']

        modules_text = []

        for mod in course_options:
            if 'footnote' in mod:
                modules_text.append(f"{mod['code']} ({mod['footnote']})")
            
            else:
                modules_text.append(mod['code'])

        content = (
            f"Group: CHS Common Core Modules\n"
            f"Pillar: {pillar_name}\n"
            f"Modules: {', '.join(modules_text)}"
        )

        chs_curr_docs.append(
                Document(
                    page_content=content,
                    metadata={"group": "CHS Common Core Modules", "pillar": pillar_name}
                )
            )

for pillars in integrated:
    pillar_name = pillars['pillar']
    course_options = pillars['course_options']
    module_codes = [mod['code'] for mod in course_options]

    content = (
            f"Group: CHS Integrated Modules\n"
            f"Pillar: {pillar_name}\n"
            f"Modules: {', '.join(module_codes)}"
        )
    
    chs_curr_docs.append(
        Document(page_content=content, metadata = {"group": "CHS Integrated Modules", "pillar": pillar_name})
    )

for pillars in interdisciplinary:
    pillar_name = pillars['pillar']
    course_options = pillars['course_options']
    module_codes = [mod['code'] for mod in course_options]

    content = (
            f"Group: CHS Interdisciplinary Modules\n"
            f"Pillar: {pillar_name}\n"
            f"Modules: {', '.join(module_codes)}"
        )
    
    chs_curr_docs.append(
        Document(page_content=content, metadata = {"group": "CHS Interdisciplinary Modules", "pillar": pillar_name})
    )
    
for student_group, preallocation in year1_preallocation.items():
    sem1_modules = preallocation['semester_1']
    sem2_modules = preallocation['semester_2']

    student = "Student ID ending with odd number" if student_group == "student_id_ending_odd" else "Student ID ending with even number"

    content = (
        f"CHS Allocated Student Group: {student}\n"
        f"Semester 1 Modules: {', '.join(sem1_modules)}\n"
        f"Semester 2 Modules {', '.join(sem2_modules)}"
    )

    chs_curr_docs.append(
        Document(page_content=content, metadata = {"group": "CHS Year 1 Preallocated Modules"})
    )

# Now we split the DSA Requirements which we have converted into a markdown file

# 1. Read markdown file
with open("data/dsa_requirements.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# 2. Split by custom separator (---)
raw_chunks = md_text.split('---')

# 3. Clean up whitespace and ignore empty chunks
chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

# 4. Wrap each chunk in a LangChain Document (optional but recommended)
dsa_docs = [Document(page_content=chunk) for chunk in chunks]

# Split NUS API Schema to allow LLM to understand the results of the API call 

def load_api_schema_docs():
  # this just gives the info needed to understand the api to help the model
    schema_text = """NUSMods module API schema essentials:

Endpoint: https://api.nusmods.com/v2/<acadYear>/modules/<moduleCode>.json

Fields:
- moduleCode, title, description, moduleCredit, faculty, department
- prerequisite, preclusion, corequisite, prerequisiteTree, fulfillRequirements
- semesterData: per-semester offerings with timetable + exam info

SemesterData entry includes:
- semester (int) and optional examDate (ISO string)
- timetable blocks (classNo, lessonType, day, startTime, endTime, weeks, venue)

Interpretation tips:
- Use semesterData to see when a module runs and lesson types offered
- Combine prerequisite + prerequisiteTree for eligibility checks
- fulfillRequirements shows downstream modules unlocked after passing
- Empty semesterData means the module is not offered that academic year

Example response for module DSA4213:
{
  "moduleCode": "DSA4213",
  "moduleCredit": "4",
  "semesterData": [
    {
      "semester": 1,
      "timetable": [
        {
          "classNo": "1",
          "lessonType": "Lecture",
          "day": "Wednesday",
          "startTime": "1000",
          "endTime": "1200",
          "weeks": [1,2,3,4,5,7,8,9,10,11,12,13],
          "venue": "COM3-01-01"
        }
      ]
    }
  ]
}
"""
    return [
        Document(
            page_content=schema_text,
            metadata={
                "source": "nusmods_api_schema",
                "category": "api_schema",
            },
        )
    ]

api_schema_docs = load_api_schema_docs()

# Merge all Documents
all_docs = chs_curr_docs + dsa_docs + api_schema_docs

# Embed and store locally
embedding = embedding()
vectorstore = FAISS.from_documents(all_docs, embedding)
vectorstore.save_local("all_requirements_vectors")

