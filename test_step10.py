import json
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
print(client.get("/").json())
client.post("/demo/seed")

summary = client.get("/class/summary").json()
for concept, info in summary["concepts"].items():
    print(concept, info)
print("weakest:", summary["weakest_concept"])

report = client.get("/students/student_03/report").json()
print(json.dumps({k: report[k] for k in ("weak_concepts", "developing", "mastered", "insufficient_evidence", "root_cause_analysis", "explanations")}, indent=2))

client.post("/responses", json={"student_id": "fresh", "concept": "loops", "correct": False, "time_seconds": 40, "attempt_number": 1})
fresh = client.get("/students/fresh/report").json()
print("fresh student ->", fresh["mastery_report"]["loops"]["status"], fresh["insufficient_evidence"])