def should_escalate(sentiment_result, threshold=0.90):
    """
    Returns True when the user's sentiment is strongly negative.

    sentiment_result is expected to be a dictionary like:
    {"label": "NEGATIVE", "score": 0.99}
    """
    label = sentiment_result.get("label", "").upper()
    score = sentiment_result.get("score", 0)

    return label == "NEGATIVE" and score > threshold


def get_escalation_message():
    """
    Returns the message shown when the user should be escalated
    to a human advisor.
    """
    return "Recommended escalation: Contact human advisor."
