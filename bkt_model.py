import math
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
    mode: str = "dampen"
    k_time: float = 0.0
    k_attempts: float = 0.0
    k_difficulty: float = 0.0


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

    def _effective_guess_slip(self, difficulty: float):
        p = self.params
        shift = p.k_difficulty * (difficulty - 0.5)
        guess = min(max(p.p_G * (1 - shift), 0.01), 0.49)
        slip = min(max(p.p_S * (1 + shift), 0.01), 0.49)
        return guess, slip

    def _feature_log_odds(self, r: ResponseInput) -> float:
        p = self.params
        value = 0.0
        if r.benchmark_time > 0 and r.time_seconds > 0:
            value -= p.k_time * (math.log(r.time_seconds / r.benchmark_time) - 0.1)
        value -= p.k_attempts * (r.attempt_number - 1)
        return max(min(value, 1.5), -1.5)

    def update(self, r: ResponseInput) -> float:
        p = self.params

        if p.mode == "likelihood":
            guess, slip = self._effective_guess_slip(r.difficulty)
            prior = min(max(self.p_L, 1e-6), 1 - 1e-6)
            odds = prior / (1 - prior)
            if r.correct:
                odds *= (1 - slip) / guess
            else:
                odds *= slip / (1 - guess)
            odds *= math.exp(self._feature_log_odds(r))
            p_L_given_obs = odds / (1 + odds)
            conf = 1.0
        else:
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

    def predict_correct(self, difficulty: float = None) -> float:
        p = self.params
        if p.mode == "likelihood" and difficulty is not None:
            guess, slip = self._effective_guess_slip(difficulty)
        else:
            guess, slip = p.p_G, p.p_S
        return self.p_L * (1 - slip) + (1 - self.p_L) * guess

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