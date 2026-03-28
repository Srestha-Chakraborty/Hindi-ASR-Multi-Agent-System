FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git build-essential \
    libenchant-2-dev pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; WhisperProcessor.from_pretrained('openai/whisper-small'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
