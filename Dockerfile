FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Minimal ffmpeg only — no git needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    opencv-python-headless \
    streamlit \
    python-dotenv \
    moviepy \
    pyyaml \
    requests \
    gdown \
    fireworks-ai

COPY . .

RUN mkdir -p data/clips

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]