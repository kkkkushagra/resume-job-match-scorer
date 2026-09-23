FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
	HF_HOME=/opt/huggingface \
	PORT=8000

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN apt-get update \
	&& apt-get install -y --no-install-recommends nodejs npm \
	&& cd frontend \
	&& npm install

COPY . .
RUN python scripts/train_resume_model.py
RUN cd frontend && npm run build

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl --fail http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
