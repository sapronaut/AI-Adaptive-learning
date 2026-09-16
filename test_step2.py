import json
from bkt_model import StudentModel, ResponseInput
from prerequisite_graph import PrerequisiteGraph

BENCHMARK_TIME = {
    "functions": 20,
    "loops": 20,
    "arrays": 22,
    "recursion": 30,
    "recursion_base_case": 30,
}

DIFFICULTY = {
    "functions": 0.3,
    "loops": 0.3,
    "arrays": 0.5,
    "recursion": 0.7,
    "recursion_base_case": 0.8,
}

PREREQ_EDGES = {
    "recursion": ["functions"],
    "recursion_base_case": ["recursion"],
    "arrays": ["loops"],
}

simulated_responses = [
    ("functions", True, 18, 1),
    ("loops", True, 19, 1),
    ("loops", True, 17, 1),
    ("arrays", False, 40, 1),
    ("arrays", False, 45, 2),
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
graph = PrerequisiteGraph(PREREQ_EDGES)
root_cause = graph.root_cause_analysis(report, threshold=0.85)

print(json.dumps({"mastery_report": report, "root_cause_analysis": root_cause}, indent=2))