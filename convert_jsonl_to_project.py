import json
import os

jsonl_path = "data/processed/labeled_dataset.jsonl"
output_path = "data/sample_project.json"

# Load the first 10 papers
papers = []
count = 0
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        papers.append({
            "id": data.get("paper_id")[:12] if data.get("paper_id") else f"paper_{count}",
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": data.get("authors", ""),
            "year": str(data.get("year")) if data.get("year") else "2025",
            "journal": data.get("journal", ""),
            "doi": data.get("doi", ""),
            "keywords": data.get("keywords", "").split(",") if data.get("keywords") else [],
            "status": "PENDING"
        })
        count += 1
        if count >= 10:
            break

# Define default criteria matching App.tsx
criteria = {
    "population": "In vivo mice or murine models (e.g. C57BL/6, ob/ob, db/db)",
    "intervention": "High fat diet HFD, high sugar diet HFHS, high cholesterol diet HCD, Western diet",
    "comparison": "Control group receiving standard chow or purified low fat diet",
    "outcome": "Metabolic outcomes: body weight, TC, TG, LDL, HDL, glucose, insulin, HOMA-IR, ALT, AST, liver steatosis",
    "studyType": "In vivo preclinical animal study with dietary intervention"
}

project_data = {
    "papers": papers,
    "criteria": criteria,
    "version": "1.0",
    "exportDate": "2026-06-04T09:00:00.000Z"
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(project_data, f, ensure_ascii=False, indent=2)

print(f"Created sample project with {len(papers)} papers at {output_path}")
