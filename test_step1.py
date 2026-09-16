import json
from bkt_model import StudentModel, ResponseInput

BENCHMARK_TIME = {
    "loops": 20,
    "arrays": 22,
    "recursion": 30,
    "recursion_base_case": 30,
}

DIFFICULTY = {
    "loops": 0.3,
    "arrays": 0.5,
    "recursion": 0.7,
    "recursion_base_case": 0.8,
}

simulated_responses = [
    ("loops", True, 20, 1),
    ("loops", True, 18, 1),
    ("arrays", True, 25, 1),
    ("arrays", False, 40, 1),
    ("recursion", False, 55, 1),
    ("recursion", False, 60, 2),
    ("recursion_base_case", False, 70, 1),
    ("recursion_base_case", False, 65, 2),
    ("recursion_base_case", True, 50, 3),
]

student = StudentModel(student_id="student_001")

for concept, correct, t, attempt in simulated_responses:
    r = ResponseInput(
        correct=correct,
        time_seconds=t,
        benchmark_time=BENCHMARK_TIME[concept],
        attempt_number=attempt,
        difficulty=DIFFICULTY[concept],
    )
    student.record_response(concept, r)

report = student.mastery_report(threshold=0.85)
weak = student.weak_concepts(threshold=0.85)

print(json.dumps({"mastery_report": report, "weak_concepts": weak}, indent=2))