
import json
from bkt_model import StudentModel, BKTParams
from gap_detector import GapDetector, ResponseEvent
from recommender import build_student_recommendations

# ---- 1. Simulated quiz response log (would come from real quiz data) ----
# Each tuple: (concept, correct, time_taken_seconds, attempt_number, hints_used)
simulated_responses = [
    ("loops", True, 20, 1, 0),
    ("loops", True, 18, 1, 0),
    ("arrays", True, 25, 1, 0),
    ("arrays", False, 40, 1, 0),
    ("recursion", False, 55, 1, 1),
    ("recursion", False, 60, 2, 1),
    ("recursion_base_case", False, 70, 1, 2),
    ("recursion_base_case", False, 65, 2, 2),
    ("recursion_base_case", True, 50, 3, 1),
]

avg_time_benchmark = {
    "loops": 20,
    "arrays": 22,
    "recursion": 30,
    "recursion_base_case": 30,
}

# ---- 2. Run through BKT model ----
student = StudentModel(student_id="student_001")
detector = GapDetector(avg_time_per_concept=avg_time_benchmark)

event_log = []
for concept, correct, t, attempt, hints in simulated_responses:
    p_mastery = student.record_response(concept, correct)
    event = ResponseEvent(concept, correct, t, attempt, hints)
    risk = detector.risk_score(p_mastery, event)
    event_log.append({
        "concept": concept, "correct": correct, "time_s": t,
        "attempt": attempt, "hints": hints,
        "p_mastery_after": round(p_mastery, 3), "risk_score": risk,
    })

# ---- 3. Mastery report + weak concept detection ----
report = student.mastery_report(threshold=0.85)
weak = student.weak_concepts(threshold=0.85)

# ---- 4. Recommendations for weak concepts ----
recommendations = build_student_recommendations(weak)

# ---- 5. Output (this is what feeds the dashboard in Phase 5) ----
output = {
    "student_id": student.student_id,
    "mastery_report": report,
    "weak_concepts": weak,
    "recommendations": recommendations,
    "event_log": event_log,
}

print(json.dumps(output, indent=2))
