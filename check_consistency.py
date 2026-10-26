import numpy as np
from bkt_model import ConceptTracker, BKTParams, ResponseInput
from real_data_experiment import forward, forward_dampen

rng = np.random.default_rng(7)
theta = (0.31, 0.12, 0.22, 0.14)
k = (1.3, 0.4, 0.8)
median_ms = 20000.0
worst = {"obs": 0.0, "damp": 0.0}

for trial in range(200):
    length = int(rng.integers(3, 40))
    y = (rng.random(length) < 0.6).astype(float)
    ms = rng.lognormal(np.log(median_ms), 0.7, length)
    n = np.where(y > 0.5, 1.0, rng.integers(1, 5, length)).astype(float)
    d = rng.random(length)
    F = {
        "Y": y[None, :], "LR": np.log(ms / median_ms)[None, :], "N": n[None, :], "D": d[None, :],
        "mask": np.ones((1, length), dtype=bool),
    }
    p_obs = forward(F, theta, k)[0]
    p_damp = forward_dampen(F, theta)[0]

    obs = ConceptTracker("c", BKTParams(p_L0=theta[0], p_T=theta[1], p_G=theta[2], p_S=theta[3], mode="likelihood", k_time=k[0], k_attempts=k[1], k_difficulty=k[2]))
    damp = ConceptTracker("c", BKTParams(p_L0=theta[0], p_T=theta[1], p_G=theta[2], p_S=theta[3]))
    for t in range(length):
        r = ResponseInput(bool(y[t]), ms[t] / 1000.0, median_ms / 1000.0, int(n[t]), float(d[t]))
        worst["obs"] = max(worst["obs"], abs(obs.predict_correct(float(d[t])) - p_obs[t]))
        worst["damp"] = max(worst["damp"], abs(damp.predict_correct() - p_damp[t]))
        obs.update(r)
        damp.update(r)

print("max abs difference, observation model:", worst["obs"])
print("max abs difference, dampening model:", worst["damp"])
print("consistent" if max(worst.values()) < 1e-9 else "MISMATCH")
