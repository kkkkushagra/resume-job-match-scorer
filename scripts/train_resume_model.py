"""Train the corpus-aware lexical model used by the FastAPI scorer."""
import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "Resume" / "Resume.csv"
OUTPUT = ROOT / "models" / "resume_tfidf.joblib"


def main() -> None:
    texts = []
    with DATASET.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            text = (row.get("Resume_str") or "").strip()
            if text:
                texts.append(text)

    if not texts:
        raise RuntimeError(f"No resume text found in {DATASET}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=100_000,
    )
    vectorizer.fit(texts)
    OUTPUT.parent.mkdir(exist_ok=True)
    joblib.dump(vectorizer, OUTPUT, compress=3)
    print(f"Trained on {len(texts)} resumes with {len(vectorizer.vocabulary_)} features")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()