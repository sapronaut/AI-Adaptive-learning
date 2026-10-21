import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from bkt_model import StudentModel, ResponseInput
from prerequisite_graph import PrerequisiteGraph
from adaptive_recommender import build_adaptive_recommendations
from explainability import build_explanations
import config
from status import classify

app = FastAPI(title="Adaptive Learning and Concept Gap Analyzer")

students: dict[str, StudentModel] = {}
graph = PrerequisiteGraph(config.PREREQ_EDGES)


class ResponseIn(BaseModel):
    student_id: str
    concept: str
    correct: bool
    time_seconds: float = Field(gt=0)
    attempt_number: int = Field(default=1, ge=1)


def get_student(student_id: str) -> StudentModel:
    if student_id not in students:
        raise HTTPException(status_code=404, detail=f"Unknown student '{student_id}'")
    return students[student_id]


def record(student_id: str, concept: str, correct: bool, time_seconds: float, attempt_number: int) -> float:
    if student_id not in students:
        students[student_id] = StudentModel(student_id)
    r = ResponseInput(
        correct=correct,
        time_seconds=time_seconds,
        benchmark_time=config.BENCHMARK_TIME[concept],
        attempt_number=attempt_number,
        difficulty=config.DIFFICULTY[concept],
    )
    return students[student_id].record_response(concept, r, config.MODEL_PARAMS)


def build_report(student: StudentModel) -> dict:
    report = student.mastery_report(config.MASTERY_THRESHOLD)
    statuses = {c: classify(info["p_mastery"], info["attempts"]) for c, info in report.items()}
    for c, status in statuses.items():
        report[c]["status"] = status

    weak = [c for c, s in statuses.items() if s == "at_risk"]
    evidence = {c: info for c, info in report.items() if info["attempts"] >= config.MIN_EVIDENCE}
    root_cause = graph.root_cause_analysis(evidence, config.AT_RISK_THRESHOLD)

    return {
        "student_id": student.student_id,
        "mastery_report": report,
        "weak_concepts": weak,
        "developing": [c for c, s in statuses.items() if s == "developing"],
        "mastered": [c for c, s in statuses.items() if s == "mastered"],
        "insufficient_evidence": [c for c, s in statuses.items() if s == "insufficient_evidence"],
        "root_cause_analysis": root_cause,
        "recommendations": build_adaptive_recommendations(weak, student.trackers),
        "explanations": build_explanations(weak, student.trackers, root_cause, config.AT_RISK_THRESHOLD),
    }


@app.get("/")
def root():
    return {
        "service": app.title,
        "students": len(students),
        "concepts": config.CONCEPTS,
        "threshold": config.MASTERY_THRESHOLD,
        "at_risk_threshold": config.AT_RISK_THRESHOLD,
        "min_evidence": config.MIN_EVIDENCE,
    }


@app.get("/dashboard")
def dashboard():
    return FileResponse(Path(__file__).parent / "dashboard.html")


@app.post("/responses")
def add_response(body: ResponseIn):
    if body.concept not in config.CONCEPTS:
        raise HTTPException(status_code=422, detail=f"Unknown concept '{body.concept}'")
    p = record(body.student_id, body.concept, body.correct, body.time_seconds, body.attempt_number)
    return {"student_id": body.student_id, "concept": body.concept, "p_mastery": round(p, 3)}


@app.get("/students")
def list_students():
    return sorted(students.keys())


@app.get("/students/{student_id}/report")
def student_report(student_id: str):
    return build_report(get_student(student_id))


@app.get("/class/summary")
def class_summary():
    if not students:
        raise HTTPException(status_code=404, detail="No students recorded yet")
    summary = {}
    for concept in config.CONCEPTS:
        values = []
        counts = {"at_risk": 0, "developing": 0, "mastered": 0, "insufficient_evidence": 0}
        for s in students.values():
            tracker = s.trackers.get(concept)
            if tracker:
                values.append(tracker.p_L)
                counts[classify(tracker.p_L, len(tracker.history))] += 1
        if values:
            summary[concept] = {
                "students": len(values),
                "mean_mastery": round(sum(values) / len(values), 3),
                "students_at_risk": counts["at_risk"],
                "students_developing": counts["developing"],
                "students_mastered": counts["mastered"],
                "students_insufficient_evidence": counts["insufficient_evidence"],
            }
    weakest = None
    if summary:
        weakest = min(summary, key=lambda c: (-summary[c]["students_at_risk"], summary[c]["mean_mastery"]))
    return {"concepts": summary, "weakest_concept": weakest}


@app.get("/class/matrix")
def class_matrix():
    return {
        "concepts": config.CONCEPTS,
        "students": {
            sid: {
                c: (round(s.trackers[c].p_L, 3) if c in s.trackers else None)
                for c in config.CONCEPTS
            }
            for sid, s in students.items()
        },
    }


@app.post("/demo/seed")
def seed_demo(num_students: int = 12, seed: int = 42):
    rng = random.Random(seed)
    students.clear()
    for i in range(num_students):
        sid = f"student_{i + 1:02d}"
        skill = rng.random()
        for concept in config.CONCEPTS:
            for attempt in range(1, rng.randint(2, 4) + 1):
                correct = rng.random() < (0.3 + 0.6 * skill)
                t = config.BENCHMARK_TIME[concept] * rng.uniform(0.7, 1.8)
                record(sid, concept, correct, t, attempt)
    return {"seeded_students": len(students)}