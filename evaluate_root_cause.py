import json
import numpy as np
import config
from bkt_model import StudentModel, ResponseInput
from api import build_report

EDGES = config.PREREQ_EDGES
ORDER = ["functions", "loops", "arrays", "recursion", "recursion_base_case"]
GUESS = 0.2
SLIP = 0.1
P_ROOT_KNOWN = 0.6
P_CHILD_KNOWN_IF_PREREQ_KNOWN = 0.8


def sample_truth(rng):
    known = {}
    for c in ORDER:
        prereqs = EDGES.get(c, [])
        if not prereqs:
            known[c] = rng.random() < P_ROOT_KNOWN
        elif all(known[p] for p in prereqs):
            known[c] = rng.random() < P_CHILD_KNOWN_IF_PREREQ_KNOWN
        else:
            known[c] = False
    return known


def truth_type(c, known):
    prereqs = EDGES.get(c, [])
    return "root_cause_upstream" if any(not known[p] for p in prereqs) else "direct_gap"


def true_roots(c, known):
    roots = []
    for p in EDGES.get(c, []):
        if not known[p]:
            roots.extend(true_roots(p, known) or [p])
    return roots


def run(n_students, n_responses, seed):
    rng = np.random.default_rng(seed)
    tp = fp = fn = tn = 0
    cls_correct = cls_total = 0
    up_total = up_hit = 0
    confusion = {"direct_gap": {"direct_gap": 0, "root_cause_upstream": 0}, "root_cause_upstream": {"direct_gap": 0, "root_cause_upstream": 0}}
    for i in range(n_students):
        known = sample_truth(rng)
        student = StudentModel(f"s{i}")
        for c in ORDER:
            for a in range(n_responses):
                p_correct = (1 - SLIP) if known[c] else GUESS
                r = ResponseInput(bool(rng.random() < p_correct), config.BENCHMARK_TIME[c], config.BENCHMARK_TIME[c], 1, config.DIFFICULTY[c])
                student.record_response(c, r, config.MODEL_PARAMS)
        report = build_report(student)
        flagged = set(report["weak_concepts"])
        for c in ORDER:
            truth_weak = not known[c]
            if truth_weak and c in flagged:
                tp += 1
            elif truth_weak:
                fn += 1
            elif c in flagged:
                fp += 1
            else:
                tn += 1
            if truth_weak and c in flagged:
                t = truth_type(c, known)
                s = report["root_cause_analysis"][c]["classification"]
                confusion[t][s] += 1
                cls_total += 1
                cls_correct += int(t == s)
                if t == "root_cause_upstream":
                    up_total += 1
                    pointed = set(report["root_cause_analysis"][c]["root_causes"])
                    if pointed & set(true_roots(c, known)):
                        up_hit += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "type_accuracy": cls_correct / cls_total if cls_total else 0.0,
        "upstream_root_hit_rate": up_hit / up_total if up_total else 0.0,
        "confusion_true_vs_system": confusion,
    }


if __name__ == "__main__":
    results = {}
    for n_resp in (3, 6, 10):
        runs = [run(300, n_resp, seed) for seed in range(10)]
        results[str(n_resp)] = {
            k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
            for k in ("precision", "recall", "f1", "type_accuracy", "upstream_root_hit_rate")
        }
        results[str(n_resp)]["confusion_true_vs_system"] = runs[0]["confusion_true_vs_system"]
        r = results[str(n_resp)]
        print(f"responses per concept {n_resp}: precision {r['precision'][0]:.3f}, recall {r['recall'][0]:.3f}, F1 {r['f1'][0]:.3f}, "
              f"direct-vs-upstream accuracy {r['type_accuracy'][0]:.3f}, upstream root hit rate {r['upstream_root_hit_rate'][0]:.3f}")
        print("   confusion (rows = truth, columns = system):", json.dumps(r["confusion_true_vs_system"]))
    with open("root_cause_results.json", "w") as f:
        json.dump(results, f, indent=2)