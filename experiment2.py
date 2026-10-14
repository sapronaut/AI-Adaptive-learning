import json
import numpy as np
from bkt_model import ConceptTracker, BKTParams, ResponseInput

CONCEPTS = ["functions", "loops", "arrays", "recursion", "recursion_base_case"]
DIFFICULTY = {"functions": 0.3, "loops": 0.3, "arrays": 0.5, "recursion": 0.7, "recursion_base_case": 0.8}
BENCH = {"functions": 20, "loops": 20, "arrays": 22, "recursion": 30, "recursion_base_case": 30}


def auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        idx = s == v
        ranks[idx] = ranks[idx].mean()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def simulate(n_students, n_items, rng, p_learn=0.15):
    data = []
    for _ in range(n_students):
        seq = []
        known = {c: rng.random() < 0.25 for c in CONCEPTS}
        for _ in range(n_items):
            c = CONCEPTS[rng.integers(len(CONCEPTS))]
            d = DIFFICULTY[c]
            guess = 0.15 + 0.1 * (1 - d)
            slip = 0.05 + 0.15 * d
            p_correct = (1 - slip) if known[c] else guess
            correct = rng.random() < p_correct
            time_factor = rng.lognormal(-0.1 if known[c] else 0.35, 0.25)
            t = BENCH[c] * time_factor
            attempt = 1 if (known[c] or rng.random() < 0.6) else int(rng.integers(2, 5))
            seq.append((c, correct, t, attempt))
            if not known[c] and rng.random() < p_learn:
                known[c] = True
        data.append(seq)
    return data


def evaluate(data, params):
    y, s = [], []
    for seq in data:
        trackers = {}
        for c, correct, t, attempt in seq:
            if c not in trackers:
                trackers[c] = ConceptTracker(c, params)
            tr = trackers[c]
            y.append(int(correct))
            s.append(tr.predict_correct(DIFFICULTY[c]))
            tr.update(ResponseInput(correct, t, BENCH[c], attempt, DIFFICULTY[c]))
    y = np.array(y)
    s = np.array(s)
    return {
        "auc": float(auc_score(y, s)),
        "accuracy": float(((s >= 0.5).astype(int) == y).mean()),
        "rmse": float(np.sqrt(np.mean((s - y) ** 2))),
    }


configs = {
    "Standard BKT (correctness only)": BKTParams(w_time=0, w_attempts=0, w_difficulty=0),
    "Dampened weighting (Step 1, all three)": BKTParams(),
    "Observation model (time only)": BKTParams(mode="likelihood", k_time=2.0),
    "Observation model (attempts only)": BKTParams(mode="likelihood", k_attempts=0.5),
    "Observation model (difficulty only)": BKTParams(mode="likelihood", k_difficulty=1.0),
    "Observation model (all three)": BKTParams(mode="likelihood", k_time=2.0, k_attempts=0.5, k_difficulty=1.0),
}

runs = 20
results = {k: {"auc": [], "accuracy": [], "rmse": []} for k in configs}
for seed in range(runs):
    rng = np.random.default_rng(seed)
    data = simulate(200, 30, rng)
    for name, params in configs.items():
        m = evaluate(data, params)
        for key in m:
            results[name][key].append(m[key])

summary = {}
for name, vals in results.items():
    summary[name] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for k, v in vals.items()}

base = np.array(results["Standard BKT (correctness only)"]["auc"])
full = np.array(results["Observation model (all three)"]["auc"])
diff = full - base
summary["paired_auc_gain_all_three_vs_standard"] = {
    "mean": round(float(diff.mean()), 4),
    "std": round(float(diff.std()), 4),
    "runs_improved": int((diff > 0).sum()),
    "runs": runs,
}

print(json.dumps(summary, indent=2))
with open("results2.json", "w") as f:
    json.dump(summary, f, indent=2)