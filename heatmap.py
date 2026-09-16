import random
import matplotlib.pyplot as plt
import numpy as np

from bkt_model import StudentModel

CONCEPTS = ["loops", "arrays", "recursion", "recursion_base_case"]
random.seed(42)


def simulate_class(num_students: int = 12):
    students = {}
    for i in range(num_students):
        sid = f"student_{i+1:02d}"
        model = StudentModel(sid)
        for concept in CONCEPTS:
            # simulate 2-4 responses per concept per student
            n_attempts = random.randint(2, 4)
            # give some students a higher "true skill" to vary results
            skill = random.random()
            for _ in range(n_attempts):
                correct = random.random() < (0.3 + 0.6 * skill)
                model.record_response(concept, correct)
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
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(CONCEPTS)))
    ax.set_xticklabels(CONCEPTS, rotation=30, ha="right")
    ax.set_yticks(range(len(student_ids)))
    ax.set_yticklabels(student_ids)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=7)

    ax.set_title("Class Concept Mastery Heatmap (P(mastery) per concept)")
    fig.colorbar(im, ax=ax, label="Mastery probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    students = simulate_class(num_students=12)
    matrix, student_ids = build_matrix(students)
    plot_heatmap(matrix, student_ids)
