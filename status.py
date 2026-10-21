import config


def classify(p_mastery: float, attempts: int) -> str:
    if attempts < config.MIN_EVIDENCE:
        return "insufficient_evidence"
    if p_mastery >= config.MASTERY_THRESHOLD:
        return "mastered"
    if p_mastery < config.AT_RISK_THRESHOLD:
        return "at_risk"
    return "developing"