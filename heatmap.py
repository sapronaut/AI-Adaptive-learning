import random
import matplotlib.pyplot as plt
import numpy as np
from bkt_model import StudentModel, ResponseInput

CONCEPTS = ["functions", "loops", "arrays", "recursion", "recursion_base_case"]

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

random.seed(42)


def simulate_class(num_students: int = 12):
    students = {}
    for i in range(num_students):
        sid = f"student_{i+1:02d}"
        model = StudentModel(sid)
        skill = random.random()
        for concept in CONCEPTS:
            n_attempts = random.randint(2, 4)
            for attempt in range(1, n_attempts + 1):
                correct = random.random() < (0.3 + 0.6 * skill)
                t = BENCHMARK_TIME[concept] * random.uniform(0.7, 1.8)
                r = ResponseInput(correct, t, BENCHMARK_TIME[concept], attempt, DIFFICULTY[concept])
                model.record_response(concept, r)
        students[sid] = model
    return students


def build_matrix(students: dict):
    matrix = np.zeros((len(students), len(CONCEPTS)))
    for i, (sid, model) in enumerate(students.items()):
        report = model.mastery_report()
        for j, concept in enumerate(CONCEPTS):
            matrix[i, j] = report.get(concept, {}).get("p_mastery", 0)
    return matrix, list(students.keys())


def plot_heatmap(matrix, student_ids, out_path="class_heatmap.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(CONCEPTS)))
    ax.set_xticklabels(CONCEPTS, rotation=30, ha="right")
    ax.set_yticks(range(len(student_ids)))
    ax.set_yticklabels(student_ids)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)

    ax.set_title("Class Concept Mastery Heatmap (Weighted BKT)")
    fig.colorbar(im, ax=ax, label="Mastery probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    students = simulate_class(num_students=12)
    matrix, student_ids = build_matrix(students)
    plot_heatmap(matrix, student_ids)