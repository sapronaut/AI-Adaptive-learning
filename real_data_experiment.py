import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

DATA = "as.csv"
TOP_SKILLS = 8
MIN_STUDENTS = 150
MAX_LEN = 100
MIN_LEN = 3
TEST_FRACTION = 0.3
HAND_K = (2.0, 0.5, 1.0)
DAMPEN_W = (0.15, 0.15, 0.2)
SHRINK = 5.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def unpack(x):
    return (
        0.01 + 0.98 * sigmoid(x[0]),
        0.001 + 0.599 * sigmoid(x[1]),
        0.01 + 0.48 * sigmoid(x[2]),
        0.01 + 0.48 * sigmoid(x[3]),
    )


def pack(theta):
    L0, T, G, S = theta
    return np.array([
        logit((L0 - 0.01) / 0.98),
        logit((T - 0.001) / 0.599),
        logit((G - 0.01) / 0.48),
        logit((S - 0.01) / 0.48),
    ])


def load(path=DATA):
    cols = ["order_id", "user_id", "problem_id", "skill_name", "correct", "attempt_count", "ms_first_response"]
    df = pd.read_csv(path, usecols=cols, low_memory=False, encoding_errors="replace")
    stats = {"raw_rows": int(len(df))}
    df = df.dropna(subset=["skill_name"])
    stats["after_missing_skill"] = int(len(df))
    df = df[df.ms_first_response > 0]
    stats["after_invalid_time"] = int(len(df))
    df = df.sort_values("order_id", kind="mergesort")
    df = df.drop_duplicates(["user_id", "problem_id", "skill_name"], keep="first")
    stats["after_duplicates"] = int(len(df))
    return df, stats


def pick_skills(df):
    g = df.groupby("skill_name").agg(rows=("correct", "size"), students=("user_id", "nunique"))
    g = g[g.students >= MIN_STUDENTS].sort_values("rows", ascending=False)
    return list(g.index[:TOP_SKILLS])


def build_sequences(df, skills):
    out = {}
    for sk in skills:
        d = df[df.skill_name == sk]
        users, ys, mss, ns, ps = [], [], [], [], []
        for u, g in d.groupby("user_id", sort=False):
            g = g.iloc[:MAX_LEN]
            if len(g) < MIN_LEN:
                continue
            users.append(u)
            ys.append(g.correct.to_numpy(float))
            mss.append(g.ms_first_response.to_numpy(float))
            ns.append(np.maximum(g.attempt_count.to_numpy(float), 1.0))
            ps.append(g.problem_id.to_numpy())
        out[sk] = {"users": np.array(users), "y": ys, "ms": mss, "n": ns, "p": ps}
    return out


def pad(seqs, fill):
    width = max(len(s) for s in seqs)
    arr = np.full((len(seqs), width), fill, dtype=float)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr


def pad_obj(seqs):
    width = max(len(s) for s in seqs)
    arr = np.full((len(seqs), width), -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr


def make_split(seq, train_users):
    is_train = np.isin(seq["users"], list(train_users))
    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]

    def take(idx):
        return {k: ([seq[k][i] for i in idx] if k != "users" else seq[k][idx]) for k in seq}

    return take(train_idx), take(test_idx)


def difficulty_table(train):
    p = np.concatenate(train["p"])
    y = np.concatenate(train["y"])
    skill_inc = float(1.0 - y.mean())
    df = pd.DataFrame({"p": p, "y": y})
    g = df.groupby("p").y.agg(["sum", "count"])
    inc = g["count"] - g["sum"]
    rate = (inc + SHRINK * skill_inc) / (g["count"] + SHRINK)
    return rate, skill_inc


def featurize(part, median_ms, rate, skill_inc):
    Y = pad(part["y"], 0.0)
    MS = pad(part["ms"], median_ms)
    N = pad(part["n"], 1.0)
    P = pad_obj(part["p"])
    lengths = np.array([len(s) for s in part["y"]])
    mask = np.arange(Y.shape[1])[None, :] < lengths[:, None]
    LR = np.log(MS / median_ms)
    D = pd.Series(P.ravel()).map(rate).fillna(skill_inc).to_numpy().reshape(P.shape)
    return {"Y": Y, "LR": LR, "N": N, "D": D, "mask": mask, "users": part["users"]}


def forward(F, theta, k=(0.0, 0.0, 0.0)):
    L0, T, G, S = theta
    kt, kn, kd = k
    Y, LR, N, D, mask = F["Y"], F["LR"], F["N"], F["D"], F["mask"]
    rows, width = Y.shape
    L = np.full(rows, L0)
    preds = np.empty((rows, width))
    for t in range(width):
        shift = kd * (D[:, t] - 0.5)
        g = np.clip(G * (1 - shift), 0.01, 0.49)
        s = np.clip(S * (1 + shift), 0.01, 0.49)
        preds[:, t] = L * (1 - s) + (1 - L) * g
        prior = np.clip(L, 1e-6, 1 - 1e-6)
        odds = prior / (1 - prior)
        odds = odds * np.where(Y[:, t] > 0.5, (1 - s) / g, s / (1 - g))
        f = np.clip(-kt * (LR[:, t] - 0.1) - kn * (N[:, t] - 1.0), -1.5, 1.5)
        odds = odds * np.exp(f)
        post = odds / (1 + odds)
        L = np.where(mask[:, t], post + (1 - post) * T, L)
    return preds


def forward_dampen(F, theta, w=DAMPEN_W):
    L0, T, G, S = theta
    wt, wa, wd = w
    Y, LR, N, D, mask = F["Y"], F["LR"], F["N"], F["D"], F["mask"]
    rows, width = Y.shape
    L = np.full(rows, L0)
    preds = np.empty((rows, width))
    for t in range(width):
        preds[:, t] = L * (1 - S) + (1 - L) * G
        ratio = np.exp(LR[:, t])
        conf = 1.0 - wt * np.minimum(np.maximum(ratio - 1.3, 0.0), 1.0) - wa * np.minimum(N[:, t] - 1.0, 3.0) / 3.0 - wd * (D[:, t] - 0.5)
        conf = np.clip(conf, 0.1, 1.0)
        correct = Y[:, t] > 0.5
        num = np.where(correct, L * (1 - S), L * S)
        den = num + np.where(correct, (1 - L) * G, (1 - L) * (1 - G))
        post = np.where(den > 0, num / den, L)
        post = L + conf * (post - L)
        L = np.where(mask[:, t], post + (1 - post) * T, L)
    return preds


def nll(F, preds):
    p = np.clip(preds[F["mask"]], 1e-4, 1 - 1e-4)
    y = F["Y"][F["mask"]]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_skill(F, k, starts):
    best = None
    for theta0 in starts:
        res = minimize(lambda x: nll(F, forward(F, unpack(x), k)), pack(theta0), method="L-BFGS-B", options={"maxiter": 25})
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x)


def fit_k(feats, thetas, k_starts, free=(True, True, True)):
    idx = [i for i in range(3) if free[i]]

    def expand(v):
        k = [0.0, 0.0, 0.0]
        for j, i in enumerate(idx):
            k[i] = float(v[j])
        return tuple(k)

    def total(v):
        k = expand(v)
        loss, count = 0.0, 0
        for sk, F in feats.items():
            n = int(F["mask"].sum())
            loss += nll(F, forward(F, thetas[sk], k)) * n
            count += n
        return loss / count

    best = None
    for k0 in k_starts:
        res = minimize(total, np.array([k0[i] for i in idx], dtype=float), method="L-BFGS-B", options={"maxiter": 25})
        if best is None or res.fun < best.fun:
            best = res
    return expand(best.x)


def fit_variant(feats, theta0, free):
    zeros = (0.0, 0.0, 0.0)
    k1 = fit_k(feats, theta0, [HAND_K, zeros], free)
    theta1 = {sk: fit_skill(F, k1, [theta0[sk]]) for sk, F in feats.items()}
    k2 = fit_k(feats, theta1, [k1], free)
    return theta1, k2


def auc(y, s):
    ranks = rankdata(s)
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pooled(feats_test, preds_by_skill):
    ys, ps, us, sks = [], [], [], []
    for sk, F in feats_test.items():
        m = F["mask"]
        ys.append(F["Y"][m])
        ps.append(preds_by_skill[sk][m])
        us.append(np.repeat(F["users"], m.sum(axis=1)))
        sks.append(np.full(int(m.sum()), sk, dtype=object))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(us), np.concatenate(sks)


def metrics(y, p):
    return {
        "auc": auc(y, p),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "accuracy": float(((p >= 0.5).astype(float) == y).mean()),
    }


def bootstrap_diff(y, p_a, p_b, users, reps, rng):
    uniq = np.unique(users)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    starts = np.searchsorted(sorted_users, uniq, side="left")
    ends = np.searchsorted(sorted_users, uniq, side="right")
    groups = [order[s:e] for s, e in zip(starts, ends)]
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        diffs.append(auc(y[idx], p_b[idx]) - auc(y[idx], p_a[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_split(seqs, seed, boot_reps):
    rng = np.random.default_rng(seed)
    all_users = np.unique(np.concatenate([s["users"] for s in seqs.values()]))
    rng.shuffle(all_users)
    n_test = int(len(all_users) * TEST_FRACTION)
    test_users = set(all_users[:n_test].tolist())
    train_users = set(all_users[n_test:].tolist())

    feats_train, feats_test, skill_means = {}, {}, {}
    for sk, seq in seqs.items():
        train, test = make_split(seq, train_users)
        median_ms = float(np.median(np.concatenate(train["ms"])))
        rate, skill_inc = difficulty_table(train)
        feats_train[sk] = featurize(train, median_ms, rate, skill_inc)
        feats_test[sk] = featurize(test, median_ms, rate, skill_inc)
        skill_means[sk] = float(np.concatenate(train["y"]).mean())

    base_start = [(0.3, 0.1, 0.2, 0.1), (0.6, 0.2, 0.1, 0.2)]
    theta0 = {sk: fit_skill(F, (0, 0, 0), base_start) for sk, F in feats_train.items()}
    variants = {
        "Observation model, fitted (all three)": (True, True, True),
        "  time only": (True, False, False),
        "  attempts only": (False, True, False),
        "  difficulty only": (False, False, True),
        "  without time": (False, True, True),
        "  without attempts": (True, False, True),
        "  without difficulty": (True, True, False),
    }
    fitted = {name: fit_variant(feats_train, theta0, free) for name, free in variants.items()}
    theta1, k2 = fitted["Observation model, fitted (all three)"]

    def predict(fn):
        return {sk: fn(sk, F) for sk, F in feats_test.items()}

    models = {
        "Skill average (no learning)": predict(lambda sk, F: np.full(F["Y"].shape, skill_means[sk])),
        "Standard BKT (fitted)": predict(lambda sk, F: forward(F, theta0[sk])),
        "Dampening weights (Step 1 design)": predict(lambda sk, F: forward_dampen(F, theta0[sk])),
        "Observation model, hand-set k": predict(lambda sk, F: forward(F, theta0[sk], HAND_K)),
    }
    for name, (th, kk) in fitted.items():
        models[name] = predict(lambda sk, F, th=th, kk=kk: forward(F, th[sk], kk))

    result = {"seed": seed, "k_fitted": {n: list(v[1]) for n, v in fitted.items()}, "models": {}, "per_skill": {}}
    pooled_cache = {}
    for name, preds in models.items():
        y, p, u, sk = pooled(feats_test, preds)
        pooled_cache[name] = (y, p, u, sk)
        result["models"][name] = metrics(y, p)
        result["models"][name]["responses"] = int(len(y))

    y, p0, u, sk = pooled_cache["Standard BKT (fitted)"]
    _, p1, _, _ = pooled_cache["Observation model, fitted (all three)"]
    mean, lo, hi = bootstrap_diff(y, p0, p1, u, boot_reps, rng)
    result["auc_gain_fitted_vs_standard"] = {"mean": mean, "ci95_low": lo, "ci95_high": hi}

    for s in seqs:
        sel = sk == s
        result["per_skill"][s] = {
            "standard": auc(y[sel], p0[sel]),
            "observation_fitted": auc(y[sel], p1[sel]),
            "test_responses": int(sel.sum()),
        }
    return result


def main(seeds, boot_reps, out):
    t0 = time.time()
    df, stats = load()
    skills = pick_skills(df)
    seqs = build_sequences(df, skills)
    summary = {
        "cleaning": stats,
        "skills": skills,
        "sequences": {s: int(len(v["users"])) for s, v in seqs.items()},
        "responses_used": int(sum(len(np.concatenate(v["y"])) for v in seqs.values())),
        "splits": [],
    }
    for seed in seeds:
        summary["splits"].append(run_split(seqs, seed, boot_reps))
        summary["elapsed_seconds"] = round(time.time() - t0, 1)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print("finished split", seed, "at", summary["elapsed_seconds"], "s", flush=True)
    return summary


if __name__ == "__main__":
    seed_list = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(seed_list, reps, "results_real.json")
