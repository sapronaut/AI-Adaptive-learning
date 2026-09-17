import json
from bkt_model import StudentModel, ResponseInput
from prerequisite_graph import PrerequisiteGraph
from adaptive_recommender import build_adaptive_recommendations
from explainability import build_explanations

PREREQ_EDGES = {
    "recursion": ["functions"],
    "recursion_base_case": ["recursion"],
    "arrays": ["loops"],
}

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
weak = student.weak_concepts(threshold=0.85)

graph = PrerequisiteGraph(PREREQ_EDGES)
root_cause = graph.root_cause_analysis(report, threshold=0.85)

recommendations = build_adaptive_recommendations(weak, student.trackers)
explanations = build_explanations(weak, student.trackers, root_cause, threshold=0.85)

output = {
    "student_id": student.student_id,
    "mastery_report": report,
    "weak_concepts": weak,
    "root_cause_analysis": root_cause,
    "recommendations": recommendations,
    "explanations": explanations,
}

print(json.dumps(output, indent=2))