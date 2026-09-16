class PrerequisiteGraph:
    def __init__(self, edges: dict[str, list[str]]):
        self.edges = edges

    def prerequisites_of(self, concept: str) -> list[str]:
        return self.edges.get(concept, [])

    def root_cause_analysis(self, mastery_report: dict, threshold: float = 0.85) -> dict:
        result = {}
        for concept, info in mastery_report.items():
            if info["p_mastery"] >= threshold:
                continue

            prereqs = self.prerequisites_of(concept)
            weak_prereqs = [
                p for p in prereqs
                if p in mastery_report and mastery_report[p]["p_mastery"] < threshold
            ]

            classification = "root_cause_upstream" if weak_prereqs else "direct_gap"

            result[concept] = {
                "classification": classification,
                "weak_prerequisites": weak_prereqs,
            }
        return result