"""
Resume <-> job description similarity scoring.

Two methods are provided:
  - semantic_similarity_score: sentence-embedding cosine similarity
    (more accurate, understands meaning/synonyms, ~90MB model download
    on first run).
  - tfidf_similarity_score: classic TF-IDF cosine similarity (fast,
    zero model download, good lightweight fallback).
"""
from functools import lru_cache
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resume_tfidf.joblib"
logger = logging.getLogger("resume_match")


@lru_cache(maxsize=1)
def _load_embedding_model():
    """Load and cache the sentence embedding model (downloaded once)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded and cached: %s", MODEL_NAME)
    return model


def semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    """0-100 semantic match score using sentence-embedding cosine similarity."""
    model = _load_embedding_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_numpy=True, normalize_embeddings=True)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    sim = max(0.0, min(1.0, sim))
    return float(round(sim * 100, 1))


@lru_cache(maxsize=1)
def _load_trained_tfidf():
    if TFIDF_MODEL_PATH.exists():
        return joblib.load(TFIDF_MODEL_PATH)
    return None


def trained_tfidf_similarity_score(resume_text: str, jd_text: str) -> float:
    """0-100 lexical score using the vectorizer trained on the resume corpus."""
    vectorizer = _load_trained_tfidf()
    if vectorizer is None:
        return tfidf_similarity_score(resume_text, jd_text)
    vectors = vectorizer.transform([resume_text, jd_text])
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return float(round(max(0.0, min(1.0, sim)) * 100, 1))


def tfidf_similarity_score(resume_text: str, jd_text: str) -> float:
    """0-100 lexical match score using TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
    except ValueError:
        # Happens if both texts are empty / contain only stop words
        return 0.0
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    sim = max(0.0, min(1.0, sim))
    return float(round(sim * 100, 1))


def similarity_scores(resume_texts: list[str], jd_text: str, method: str = "semantic") -> list[float]:
    """Score a batch together to avoid repeating model and vector operations."""
    if not resume_texts:
        return []
    if method == "tfidf":
        vectorizer = _load_trained_tfidf()
        if vectorizer is None:
            return [tfidf_similarity_score(text, jd_text) for text in resume_texts]
        vectors = vectorizer.transform([*resume_texts, jd_text])
        similarities = cosine_similarity(vectors[:-1], vectors[-1:]).ravel()
        return [float(round(max(0.0, min(1.0, score)) * 100, 1)) for score in similarities]

    model = _load_embedding_model()
    embeddings = model.encode([*resume_texts, jd_text], convert_to_numpy=True, normalize_embeddings=True)
    similarities = np.dot(embeddings[:-1], embeddings[-1])
    semantic_scores = [max(0.0, min(1.0, score)) * 100 for score in similarities]
    vectorizer = _load_trained_tfidf()
    if vectorizer is None:
        return [float(round(score, 1)) for score in semantic_scores]
    vectors = vectorizer.transform([*resume_texts, jd_text])
    lexical_scores = cosine_similarity(vectors[:-1], vectors[-1:]).ravel() * 100
    return [
        float(round((semantic_score * 0.8) + (max(0.0, min(100.0, lexical_score)) * 0.2), 1))
        for semantic_score, lexical_score in zip(semantic_scores, lexical_scores)
    ]


def similarity_score(resume_text: str, jd_text: str, method: str = "semantic") -> float:
    """Unified entry point. method: 'semantic' or 'tfidf'."""
    return similarity_scores([resume_text], jd_text, method=method)[0]
