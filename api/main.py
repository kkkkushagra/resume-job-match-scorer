from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.parser import extract_text
from utils.evidence import analyze_resume_match
from utils.skills import load_taxonomy, skill_gap_analysis


app = FastAPI(title="Resume Job Match API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TAXONOMY = load_taxonomy()
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def validate_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Every uploaded file needs a filename.")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Use PDF, DOCX, or TXT files only.")
    return filename


async def read_upload(upload: UploadFile) -> str:
    filename = validate_filename(upload.filename)
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    try:
        text = extract_text(BytesIO(content), filename=filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {filename}: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail=f"Could not extract text from {filename}.")
    return text


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/score")
async def score_resumes(
    jd_text: Annotated[str, Form()] = "",
    jd_file: Annotated[UploadFile | None, File()] = None,
    resumes: Annotated[list[UploadFile] | None, File()] = None,
    method: Annotated[str, Form()] = "semantic",
) -> dict:
    if jd_file is not None:
        jd_text = await read_upload(jd_file)
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Provide a job description as text or a file.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")
    if method not in {"semantic", "tfidf"}:
        raise HTTPException(status_code=400, detail="Scoring method must be semantic or tfidf.")

    resume_names = []
    resume_texts = []
    for resume in resumes:
        filename = validate_filename(resume.filename)
        resume_names.append(filename)
        resume_texts.append(await read_upload(resume))

    results = []
    for filename, resume_text in zip(resume_names, resume_texts):
        analysis = analyze_resume_match(resume_text, jd_text, method=method, taxonomy=TAXONOMY)
        matched, missing = skill_gap_analysis(resume_text, jd_text, TAXONOMY)
        results.append(
            {
                "filename": filename,
                "score": analysis["overall_score"],
                "matched": matched,
                "missing": missing,
                "analysis": analysis,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return {"results": results, "method": method}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        requested = FRONTEND_DIST / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
