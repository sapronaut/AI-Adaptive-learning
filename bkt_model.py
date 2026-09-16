from dataclasses import dataclass


@dataclass
class BKTParams:
    p_L0: float = 0.3
    p_T: float = 0.15
    p_G: float = 0.2
    p_S: float = 0.1
    w_time: float = 0.15
    w_attempts: float = 0.15
    w_difficulty: float = 0.2


@dataclass
class ResponseInput:
    correct: bool
    time_seconds: float
    benchmark_time: float
    attempt_number: int
    difficulty: float


class ConceptTracker:
    def __init__(self, concept_name: str, params: BKTParams = None):
        self.concept_name = concept_name
        self.params = params or BKTParams()
        self.p_L = self.params.p_L0
        self.history = []

    def _evidence_weight(self, r: ResponseInput) -> float:
        p = self.params
        weight = 1.0

        if r.benchmark_time > 0:
            ratio = r.time_seconds / r.benchmark_time
            if ratio > 1.3:
                weight -= p.w_time * min(ratio - 1.3, 1.0)

        if r.attempt_number > 1:
            weight -= p.w_attempts * min(r.attempt_number - 1, 3) / 3

        weight -= p.w_difficulty * (r.difficulty - 0.5)

        return max(min(weight, 1.0), 0.1)

    def update(self, r: ResponseInput) -> float:
        p = self.params
        conf = self._evidence_weight(r)

        if r.correct:
            numerator = self.p_L * (1 - p.p_S)
            denominator = numerator + (1 - self.p_L) * p.p_G
        else:
            numerator = self.p_L * p.p_S
            denominator = numerator + (1 - self.p_L) * (1 - p.p_G)

        p_L_given_obs = numerator / denominator if denominator > 0 else self.p_L
        p_L_given_obs = self.p_L + conf * (p_L_given_obs - self.p_L)

        self.p_L = p_L_given_obs + (1 - p_L_given_obs) * p.p_T
        self.history.append((r, self.p_L, conf))
        return self.p_L

    def is_mastered(self, threshold: float = 0.85) -> bool:
        return self.p_L >= threshold


class StudentModel:
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.trackers: dict[str, ConceptTracker] = {}

    def record_response(self, concept: str, r: ResponseInput, params: BKTParams = None):
        if concept not in self.trackers:
            self.trackers[concept] = ConceptTracker(concept, params)
        return self.trackers[concept].update(r)

    def mastery_report(self, threshold: float = 0.85) -> dict:
        return {
            concept: {
                "p_mastery": round(tracker.p_L, 3),
                "mastered": tracker.is_mastered(threshold),
                "attempts": len(tracker.history),
            }
            for concept, tracker in self.trackers.items()
        }

    def weak_concepts(self, threshold: float = 0.85) -> list[str]:
        return [c for c, t in self.trackers.items() if not t.is_mastered(threshold)]