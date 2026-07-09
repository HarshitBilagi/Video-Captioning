FROM python:3.11-slim

# Install system dependencies for OpenCV and moviepy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import whisper; whisper.load_model('base')"

COPY . .

CMD ["streamlit", "run", "app.py"]
