import json
import sys
import numpy as np
from real_data_experiment import (
    load, pick_skills, build_sequences, make_split, difficulty_table,
    featurize, fit_skill, fit_variant, auc, TEST_FRACTION,
)

MASTERY = 0.85
AT_RISK = 0.5
STRUGGLE_BELOW = 0.6
MIN_FUTURE = 5
EARLY_COUNTS = (3, 5, 8)


def states(F, theta, k):
    L0, T, G, S = theta
    kt, kn, kd = k
    Y, LR, N, D, mask = F["Y"], F["LR"], F["N"], F["D"], F["mask"]
    rows, width = Y.shape
    L = np.full(rows, L0)
    after = np.empty((rows, width))
    for t in range(width):
        shift = kd * (D[:, t] - 0.5)
        g = np.clip(G * (1 - shift), 0.01, 0.49)
        s = np.clip(S * (1 + shift), 0.01, 0.49)
        prior = np.clip(L, 1e-6, 1 - 1e-6)
        odds = prior / (1 - prior)
        odds = odds * np.where(Y[:, t] > 0.5, (1 - s) / g, s / (1 - g))
        f = np.clip(-kt * (LR[:, t] - 0.1) - kn * (N[:, t] - 1.0), -1.5, 1.5)
        odds = odds * np.exp(f)
        post = odds / (1 + odds)
        L = np.where(mask[:, t], post + (1 - post) * T, L)
        after[:, t] = L
    return after


def status_of(p):
    return np.where(p >= MASTERY, "mastered", np.where(p < AT_RISK, "at_risk", "developing"))


def run_split(seqs, seed):
    rng = np.random.default_rng(seed)
    users = np.unique(np.concatenate([s["users"] for s in seqs.values()]))
    rng.shuffle(users)
    n_test = int(len(users) * TEST_FRACTION)
    test_users, train_users = set(users[:n_test].tolist()), set(users[n_test:].tolist())

    feats_train, feats_test = {}, {}
    for sk, seq in seqs.items():
        tr, te = make_split(seq, train_users)
        med = float(np.median(np.concatenate(tr["ms"])))
        rate, sinc = difficulty_table(tr)
        feats_train[sk] = featurize(tr, med, rate, sinc)
        feats_test[sk] = featurize(te, med, rate, sinc)

    theta0 = {sk: fit_skill(F, (0, 0, 0), [(0.3, 0.1, 0.2, 0.1), (0.6, 0.2, 0.1, 0.2)]) for sk, F in feats_train.items()}
    theta1, k1 = fit_variant(feats_train, theta0, (False, False, True))

    out = {}
    for m in EARLY_COUNTS:
        rows = {"fitted": [], "standard": [], "early_acc": [], "future": [], "n_future": []}
        for sk, F in feats_test.items():
            lengths = F["mask"].sum(axis=1)
            keep = lengths >= m + MIN_FUTURE
            if not keep.any():
                continue
            fit_L = states(F, theta1[sk], k1)[:, m - 1][keep]
            std_L = states(F, theta0[sk], (0.0, 0.0, 0.0))[:, m - 1][keep]
            Y = F["Y"][keep]
            ln = lengths[keep]
            early = Y[:, :m].mean(axis=1)
            fut = np.array([Y[i, m:ln[i]].mean() for i in range(len(ln))])
            rows["fitted"].append(fit_L)
            rows["standard"].append(std_L)
            rows["early_acc"].append(early)
            rows["future"].append(fut)
        fitted = np.concatenate(rows["fitted"])
        standard = np.concatenate(rows["standard"])
        early = np.concatenate(rows["early_acc"])
        future = np.concatenate(rows["future"])
        struggle = (future < STRUGGLE_BELOW).astype(float)

        status = status_of(fitted)
        groups = {}
        for name in ("mastered", "developing", "at_risk"):
            sel = status == name
            groups[name] = {
                "n": int(sel.sum()),
                "share": float(sel.mean()),
                "future_accuracy": float(future[sel].mean()) if sel.any() else None,
                "struggle_rate": float(struggle[sel].mean()) if sel.any() else None,
            }
        flagged = status == "at_risk"
        precision = float(struggle[flagged].mean()) if flagged.any() else None
        recall = float(flagged[struggle > 0.5].mean()) if (struggle > 0.5).any() else None
        out[str(m)] = {
            "students_skills": int(len(future)),
            "struggle_base_rate": float(struggle.mean()),
            "auc_fitted_model": auc(struggle, 1 - fitted),
            "auc_standard_bkt": auc(struggle, 1 - standard),
            "auc_early_accuracy": auc(struggle, 1 - early),
            "groups": groups,
            "at_risk_precision": precision,
            "at_risk_recall": recall,
        }
    return {"seed": seed, "k": list(k1), "by_early_count": out}


def main(seeds, out_path):
    df, _ = load()
    seqs = build_sequences(df, pick_skills(df))
    results = {"definitions": {
        "struggling": "accuracy on later responses below " + str(STRUGGLE_BELOW),
        "min_future_responses": MIN_FUTURE,
        "mastered_at": MASTERY,
        "at_risk_below": AT_RISK,
    }, "splits": []}
    for seed in seeds:
        results["splits"].append(run_split(seqs, seed))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("finished split", seed, flush=True)
    return results


if __name__ == "__main__":
    seed_list = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    main(seed_list, "gap_detection_results.json")