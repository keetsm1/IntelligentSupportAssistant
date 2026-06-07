from transformers import pipeline

# Loaded once at import time so repeated calls in the main loop stay fast.
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text):
    """
    Runs sentiment analysis on the given text.

    Returns a dict like {"label": "POSITIVE", "score": 0.999} so callers
    can access both the predicted sentiment and the model's confidence.
    """
    result = _sentiment_pipeline(text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"])
    }
