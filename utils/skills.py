"""
Rule-based skill extraction and skill-gap analysis.

Uses a curated skills taxonomy (data/skills_taxonomy.json) and phrase
matching to answer: which skills does the JD ask for, which of those
does the resume actually mention, and what's missing (the likely
reason a resume scores low).
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_taxonomy.json"

# Collapse near-duplicate phrasings onto one canonical skill name so the
# same skill isn't reported twice (once per phrasing) in the results.
ALIASES = {
    "Natural Language Processing": "NLP",
    "Google Cloud Platform": "GCP",
    "Applicant Tracking System": "ATS",
}


def load_taxonomy() -> Dict[str, List[str]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(taxonomy: Dict[str, List[str]]) -> List[str]:
    skills = []
    for category_skills in taxonomy.values():
        skills.extend(category_skills)
    # Longer phrases first, so e.g. "Machine Learning" is checked before "Machine"
    return sorted(set(skills), key=len, reverse=True)


def _build_pattern(skill: str) -> re.Pattern:
    # Case-insensitive match with loose word boundaries that still allow
    # skills containing +, #, or . (e.g. "C++", "C#", "Node.js").
    escaped = re.escape(skill)
    return re.compile(rf"(?<![\w+#.]){escaped}(?![\w+#.])", re.IGNORECASE)


_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def extract_skills(text: str, taxonomy: Dict[str, List[str]] = None) -> Set[str]:
    """Return the set of taxonomy skills found in `text`."""
    if taxonomy is None:
        taxonomy = load_taxonomy()
    text = text or ""
    found = set()
    for skill in _flatten(taxonomy):
        pattern = _PATTERN_CACHE.get(skill)
        if pattern is None:
            pattern = _build_pattern(skill)
            _PATTERN_CACHE[skill] = pattern
        if pattern.search(text):
            found.add(ALIASES.get(skill, skill))
    return found


def skill_gap_analysis(
    resume_text: str, jd_text: str, taxonomy: Dict[str, List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Compare skills mentioned in the JD against skills found in the resume.
    Returns (matched_skills, missing_skills) as sorted lists.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy()
    jd_skills = extract_skills(jd_text, taxonomy)
    resume_skills = extract_skills(resume_text, taxonomy)

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    return matched, missing
