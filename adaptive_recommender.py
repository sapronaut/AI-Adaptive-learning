RESOURCE_BANK = {
    "functions": {
        "recovering": [{"type": "practice", "title": "Function Practice Set B", "difficulty": "medium"}],
        "stagnant": [{"type": "video", "title": "Functions Re-explained", "difficulty": "medium"}],
        "declining": [{"type": "video", "title": "Functions from Scratch", "difficulty": "easy"}],
    },
    "loops": {
        "recovering": [{"type": "practice", "title": "Loop Warm-up Drills", "difficulty": "easy"}],
        "stagnant": [{"type": "video", "title": "Loops Re-explained Visually", "difficulty": "medium"}],
        "declining": [{"type": "video", "title": "Loops from Scratch", "difficulty": "easy"}],
    },
    "arrays": {
        "recovering": [{"type": "practice", "title": "Array Manipulation Set B", "difficulty": "medium"}],
        "stagnant": [{"type": "video", "title": "Arrays: Common Pitfalls", "difficulty": "medium"}],
        "declining": [{"type": "video", "title": "Array Fundamentals Reset", "difficulty": "easy"}],
    },
    "recursion": {
        "recovering": [{"type": "practice", "title": "Recursion Practice Set B", "difficulty": "medium"}],
        "stagnant": [{"type": "video", "title": "Recursion Explained Visually", "difficulty": "medium"}],
        "declining": [{"type": "video", "title": "Recursion from First Principles", "difficulty": "easy"}],
    },
    "recursion_base_case": {
        "recovering": [{"type": "practice", "title": "Base Case Debugging Set B", "difficulty": "medium"}],
        "stagnant": [{"type": "video", "title": "Why Base Cases Matter", "difficulty": "medium"}],
        "declining": [{"type": "video", "title": "Base Cases: Start Here", "difficulty": "easy"}],
    },
}


def infer_learner_state(history: list) -> str:
    if len(history) < 2:
        return "stagnant"

    recent = [p_L for _, p_L, _ in history[-3:]]
    trend = recent[-1] - recent[0]

    if trend > 0.1:
        return "recovering"
    if trend < -0.05:
        return "declining"
    return "stagnant"


def recommend_for_concept(concept: str, learner_state: str, max_items: int = 2):
    bank = RESOURCE_BANK.get(concept, {})
    items = bank.get(learner_state, bank.get("stagnant", []))
    if not items:
        items = [{"type": "note", "title": f"No curated resources for '{concept}'", "difficulty": "n/a"}]
    return items[:max_items]


def build_adaptive_recommendations(weak_concepts: list[str], trackers: dict) -> dict:
    recs = {}
    for concept in weak_concepts:
        tracker = trackers.get(concept)
        state = infer_learner_state(tracker.history) if tracker else "stagnant"
        recs[concept] = {
            "learner_state": state,
            "resources": recommend_for_concept(concept, state),
        }
    return recs