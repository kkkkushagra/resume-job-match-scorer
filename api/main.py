from io import BytesIO
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.parser import extract_text
from utils.evidence import analyze_resume_match, prepare_requirement_context
from utils.scorer import _load_embedding_model
from utils.skills import load_taxonomy, skill_gap_analysis


logger = logging.getLogger("resume_match")
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


@app.on_event("startup")
def warm_embedding_model() -> None:
    try:
        _load_embedding_model()
        logger.info("Embedding model warmup completed")
    except Exception:
        logger.exception("Embedding model warmup failed; semantic scoring may be unavailable")


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
        logger.exception("Upload parsing failed: filename=%s size=%d", filename, len(content))
        raise HTTPException(status_code=400, detail=f"Could not read {filename}: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail=f"Could not extract text from {filename}.")
    return text


@app.exception_handler(Exception)
async def handle_unexpected_exception(request, exc: Exception):
    logger.exception("Unhandled request failure: method=%s path=%s", request.method, request.url.path)
    safe_detail = f"{type(exc).__name__}: {str(exc)[:300]}".strip()
    if request.url.path == "/api/score":
        return JSONResponse(
            status_code=500,
            content={"error": "Scoring failed", "detail": safe_detail or "The scoring pipeline failed. Check the Render logs for the traceback."},
        )
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": safe_detail})


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
    logger.info("Scoring request started: method=%s resumes=%d", method, len(resumes or []))
    if jd_file is not None:
        jd_text = await read_upload(jd_file)
        logger.info("Job description parsed from upload: characters=%d", len(jd_text))
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
        resume_text = await read_upload(resume)
        resume_texts.append(resume_text)
        logger.info("Resume parsed: filename=%s characters=%d", filename, len(resume_text))

    logger.info("Preparing requirement embeddings: requirements from %d job-description characters", len(jd_text))
    requirements, requirement_embeddings = prepare_requirement_context(
        jd_text,
        method=method,
        taxonomy=TAXONOMY,
    )
    results = []
    for filename, resume_text in zip(resume_names, resume_texts):
        logger.info("Analyzing resume: filename=%s", filename)
        analysis = analyze_resume_match(
            resume_text,
            jd_text,
            method=method,
            taxonomy=TAXONOMY,
            requirements=requirements,
            requirement_embeddings=requirement_embeddings,
        )
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
    logger.info("Scoring request completed: results=%d", len(results))
    return {"results": results, "method": method}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        requested = FRONTEND_DIST / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
