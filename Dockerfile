FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN apt-get update \
	&& apt-get install -y --no-install-recommends nodejs npm \
	&& cd frontend \
	&& npm install

COPY . .
RUN python scripts/train_resume_model.py
RUN cd frontend && npm run build

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/api/health || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
