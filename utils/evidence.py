"""General-purpose semantic requirement-to-resume evidence matching."""
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.scorer import _load_embedding_model, _load_trained_tfidf
from utils.skills import extract_skills, load_taxonomy


REQUIREMENT_LIMIT = 24
ACTION_WORDS = re.compile(
    r"\b(build|built|develop|developed|design|designed|implement|implemented|"
    r"test|tested|deploy|deployed|manage|managed|lead|led|create|created|"
    r"analy[sz]e|maintain|maintained|deliver|delivered|automate|automated)\w*\b",
    re.IGNORECASE,
)
SECTION_WORDS = {
    "experience": ("experience", "work", "employment", "professional"),
    "projects": ("project", "portfolio", "application"),
    "internships": ("intern", "internship"),
    "education": ("education", "academic", "coursework", "degree", "university"),
    "certifications": ("certification", "certificate", "licensed"),
    "skills": ("skill", "technical", "competenc", "proficien"),
    "summary": ("summary", "objective", "profile"),
}


@dataclass
class Requirement:
    text: str
    category: str
    mandatory: bool
    preferred: bool


@dataclass
class RequirementMatch:
    requirement: str
    category: str
    mandatory: bool
    preferred: bool
    match: str
    similarity: float
    evidence_strength: str
    evidence: str
    reason: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"[•\u2022]+", "\n", text or "")
    pieces = re.split(r"\n+|(?<=[.!?])\s+", normalized)
    return [_clean_text(piece).strip("-*") for piece in pieces if len(_clean_text(piece).split()) >= 3]


def _unique(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _is_mandatory(text: str) -> bool:
    return bool(re.search(r"\b(required|must|required to|minimum|essential|mandatory|need to)\b", text, re.I))


def _is_preferred(text: str) -> bool:
    return bool(re.search(r"\b(preferred|ideally|nice to have|plus|bonus|desirable)\b", text, re.I))


def _category(text: str, taxonomy: dict[str, list[str]]) -> str:
    lower = text.lower()
    if re.search(r"\b(bachelor|master|degree|university|college|education|academic)\b", lower):
        return "Education"
    if re.search(r"\b(certif|license|licen[cs]e|credential)\w*\b", lower):
        return "Certifications"
    if re.search(r"\b(\d+\+?\s+years?|years? of experience|senior|junior|internship)\b", lower):
        return "Experience"
    if ACTION_WORDS.search(text):
        return "Responsibilities"
    if extract_skills(text, taxonomy):
        return "Technical Skills"
    if re.search(r"\b(communication|leadership|teamwork|collaborat|negotiat|presentation|adaptab)\w*\b", lower):
        return "Soft Skills"
    return "Domain Knowledge"


def extract_requirements(jd_text: str, taxonomy: dict[str, list[str]] | None = None) -> list[Requirement]:
    taxonomy = taxonomy or load_taxonomy()
    requirements = []
    for sentence in _unique(_sentences(jd_text)):
        requirements.append(
            Requirement(
                text=sentence,
                category=_category(sentence, taxonomy),
                mandatory=_is_mandatory(sentence),
                preferred=_is_preferred(sentence),
            )
        )
    return requirements[:REQUIREMENT_LIMIT]


def _resume_chunks(resume_text: str) -> list[tuple[str, str]]:
    chunks = []
    current_section = "Other"
    for line in (resume_text or "").splitlines():
        clean = _clean_text(line)
        if not clean:
            continue
        lower = clean.lower()
        detected = next((section for section, words in SECTION_WORDS.items() if any(word in lower for word in words)), None)
        if detected and len(clean.split()) <= 8:
            current_section = detected.title()
            continue
        for sentence in _sentences(clean) or [clean]:
            chunks.append((sentence, current_section))
    if not chunks:
        chunks = [(sentence, "Other") for sentence in _sentences(resume_text)]
    return _unique_chunks(chunks)


def _unique_chunks(chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    result = []
    for text, section in chunks:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append((text, section))
    return result[:160]


def _evidence_factor(text: str, section: str) -> float:
    factor = {
        "Experience": 1.0,
        "Projects": 0.92,
        "Internships": 0.86,
        "Certifications": 0.78,
        "Education": 0.68,
        "Skills": 0.62,
        "Summary": 0.64,
        "Other": 0.62,
    }.get(section, 0.62)
    if ACTION_WORDS.search(text):
        factor += 0.1
    if re.search(r"\b(\d+\+?\s+years?|19\d{2}|20\d{2})\b", text, re.I):
        factor += 0.05
    return min(1.0, factor)


@lru_cache(maxsize=128)
def _encode_resume_chunks(chunks: tuple[str, ...]) -> np.ndarray:
    model = _load_embedding_model()
    return model.encode(list(chunks), convert_to_numpy=True, normalize_embeddings=True)


def _exact_skill_match(requirement: str, evidence: str, taxonomy: dict[str, list[str]]) -> bool:
    requirement_skills = extract_skills(requirement, taxonomy)
    evidence_skills = extract_skills(evidence, taxonomy)
    return bool(requirement_skills & evidence_skills)


def _classify(similarity: float, exact: bool) -> str:
    if exact:
        return "Exact match"
    if similarity >= 0.38:
        return "Strong semantic match"
    if similarity >= 0.27:
        return "Related but not confirmed"
    if similarity >= 0.18:
        return "Partial match"
    return "Not found"


def _reason(match: str, requirement: str, evidence: str) -> str:
    if match == "Exact match":
        return "The resume explicitly demonstrates the requested skill or technology."
    if match == "Strong semantic match":
        return "The resume evidence describes closely related work, even without the same wording."
    if match == "Related but not confirmed":
        return "The evidence is conceptually related, but direct experience with the requirement is not confirmed."
    if match == "Partial match":
        return "The evidence overlaps with part of the requirement but does not fully demonstrate it."
    return "No sufficiently relevant resume evidence was found for this requirement."


def analyze_resume_match(
    resume_text: str,
    jd_text: str,
    method: str = "semantic",
    taxonomy: dict[str, list[str]] | None = None,
) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    requirements = extract_requirements(jd_text, taxonomy)
    chunks = _resume_chunks(resume_text)
    if not requirements or not chunks:
        return {"overall_score": 0.0, "requirements": [], "strong_matches": [], "partial_matches": [], "related_matches": [], "missing": [], "mandatory_missing": []}

    requirement_texts = [requirement.text for requirement in requirements]
    evidence_texts = [chunk[0] for chunk in chunks]
    if method == "tfidf" and _load_trained_tfidf() is not None:
        vectorizer = _load_trained_tfidf()
        vectors = vectorizer.transform([*requirement_texts, *evidence_texts])
        matrix = cosine_similarity(vectors[: len(requirements)], vectors[len(requirements) :])
    else:
        model = _load_embedding_model()
        requirement_embeddings = model.encode(requirement_texts, convert_to_numpy=True, normalize_embeddings=True)
        evidence_embeddings = _encode_resume_chunks(tuple(evidence_texts))
        matrix = np.dot(requirement_embeddings, evidence_embeddings.T)

    matches = []
    for index, requirement in enumerate(requirements):
        best_index = int(np.argmax(matrix[index]))
        similarity = float(max(0.0, min(1.0, matrix[index][best_index])))
        evidence, section = chunks[best_index]
        exact = _exact_skill_match(requirement.text, evidence, taxonomy)
        match = _classify(similarity, exact)
        factor = _evidence_factor(evidence, section)
        evidence_strength = "Demonstrated" if factor >= 0.85 else "Practical" if factor >= 0.75 else "Mentioned"
        matches.append(
            RequirementMatch(
                requirement=requirement.text,
                category=requirement.category,
                mandatory=requirement.mandatory,
                preferred=requirement.preferred,
                match=match,
                similarity=round(similarity * 100, 1),
                evidence_strength=evidence_strength,
                evidence=f"{evidence} [{section}]",
                reason=_reason(match, requirement.text, evidence),
            )
        )

    value_map = {"Exact match": 1.0, "Strong semantic match": 0.88, "Related but not confirmed": 0.62, "Partial match": 0.4, "Not found": 0.0}
    weighted_total = 0.0
    weight_total = 0.0
    for match in matches:
        weight = 1.25 if match.mandatory else 0.85 if match.preferred else 1.0
        factor = {"Demonstrated": 1.0, "Practical": 0.9, "Mentioned": 0.72}.get(match.evidence_strength, 0.72)
        weighted_total += value_map[match.match] * factor * weight
        weight_total += weight
    score = (weighted_total / weight_total) * 100 if weight_total else 0.0
    missing_mandatory = [asdict(match) for match in matches if match.mandatory and match.match in {"Not found", "Partial match"}]
    if missing_mandatory:
        score *= max(0.55, 1.0 - (0.1 * len(missing_mandatory)))

    serialized = [asdict(match) for match in matches]
    return {
        "overall_score": round(score, 1),
        "requirements": serialized,
        "strong_matches": [item for item in serialized if item["match"] in {"Exact match", "Strong semantic match"}],
        "partial_matches": [item for item in serialized if item["match"] == "Partial match"],
        "related_matches": [item for item in serialized if item["match"] == "Related but not confirmed"],
        "missing": [item for item in serialized if item["match"] == "Not found"],
        "mandatory_missing": missing_mandatory,
    }
