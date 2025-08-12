import re
from typing import List

def _simple_tokenize(text: str) -> list:
    return re.findall(r"[\wáéíóúñ]+", text.lower())

_POSITIVE = set("""excelente bueno buena recomendada recomiendo maravilloso rapido rápida cumplen cumplido
eficiente amable profesional calidad fresco frescos fresco(a) super súper puntuales responsable satisfecho
""".split())

_NEGATIVE = set("""malo mala tarde retraso retrasados deficiente pésimo queja quejas lento lenta caro
caros fallas problemas nunca jamas jamás
""".split())

def simple_sentiment_score(text: str) -> float:
    """Devuelve un score en [0,1] usando un léxico mínimo (fallback).
    0 = muy negativo, 1 = muy positivo.
    """
    toks = _simple_tokenize(text)
    if not toks:
        return 0.5
    pos = sum(1 for t in toks if t in _POSITIVE)
    neg = sum(1 for t in toks if t in _NEGATIVE)
    raw = 0.5 + 0.1 * (pos - neg)
    return max(0.0, min(1.0, raw))

def reviews_sentiment(reviews: List[str]) -> float:
    if not reviews:
        return 0.5
    scores = [simple_sentiment_score(r) for r in reviews]
    return sum(scores) / len(scores)

try:
    # Intento de usar transformers si está instalado
    from transformers import pipeline
    _pipeline = pipeline("sentiment-analysis")
    def reviews_sentiment(reviews: List[str]) -> float:  # override
        if not reviews:
            return 0.5
        outs = _pipeline(reviews)
        mapped = []
        for o in outs:
            label = o.get("label","").upper()
            score = float(o.get("score", 0.5))
            mapped.append(score if "POS" in label else (1.0-score if "NEG" in label else 0.5))
        return sum(mapped) / len(mapped)
except Exception:
    # Mantener fallback simple
    pass
