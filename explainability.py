def explain_concept_flag(concept: str, tracker, root_cause_info: dict, threshold: float = 0.85) -> str:
    reasons = []

    p_final = tracker.p_L
    reasons.append(f"mastery probability {p_final:.2f} is below the {threshold} threshold")

    wrong_count = sum(1 for r, _, _ in tracker.history if not r.correct)
    if wrong_count > 0:
        reasons.append(f"{wrong_count} incorrect attempt(s) recorded")

    slow_events = [
        r for r, _, _ in tracker.history
        if r.benchmark_time > 0 and r.time_seconds > 1.3 * r.benchmark_time
    ]
    if slow_events:
        reasons.append(f"{len(slow_events)} response(s) significantly slower than benchmark")

    info = root_cause_info.get(concept)
    if info and info["classification"] == "root_cause_upstream":
        prereq_list = ", ".join(info["weak_prerequisites"])
        reasons.append(f"linked to unresolved prerequisite gap(s) in: {prereq_list}")

    return f"Flagged '{concept}' because " + "; ".join(reasons) + "."


def build_explanations(weak_concepts: list[str], trackers: dict, root_cause_info: dict, threshold: float = 0.85) -> dict:
    return {
        concept: explain_concept_flag(concept, trackers[concept], root_cause_info, threshold)
        for concept in weak_concepts if concept in trackers
    }