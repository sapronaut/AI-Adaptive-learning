from bkt_model import BKTParams

CONCEPTS = ["functions", "loops", "arrays", "recursion", "recursion_base_case"]

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

MASTERY_THRESHOLD = 0.85
AT_RISK_THRESHOLD = 0.5
MIN_EVIDENCE = 2

MODEL_PARAMS = BKTParams(mode="likelihood", k_time=0.0, k_attempts=0.0, k_difficulty=1.0)