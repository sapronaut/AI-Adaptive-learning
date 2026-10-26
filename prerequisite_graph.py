class PrerequisiteGraph:
    def __init__(self, edges: dict[str, list[str]]):
        self.edges = edges

    def prerequisites_of(self, concept: str) -> list[str]:
        return self.edges.get(concept, [])

    def _roots(self, concept: str, result: dict, seen: set) -> list[str]:
        roots = []
        for p in result[concept]["weak_prerequisites"]:
            if p in seen:
                continue
            seen.add(p)
            if result[p]["classification"] == "direct_gap":
                roots.append(p)
            else:
                roots.extend(self._roots(p, result, seen))
        return list(dict.fromkeys(roots))

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
                "root_causes": [],
            }

        for concept, info in result.items():
            if info["classification"] == "root_cause_upstream":
                info["root_causes"] = self._roots(concept, result, {concept})
        return result