# Video Captioning Pipeline

This repository implements a modular video captioning pipeline designed to extract visual frames and transcriptions, perform scene understanding via vision-language models, generate captions matching specific tones, and run a self-judgment loop for quality control.

## Project Structure

```text
repo/
├── README.md
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── models.yaml
├── src/
│   ├── __init__.py
│   ├── extraction.py
│   ├── scene_understanding.py
│   ├── caption_generation.py
│   ├── self_judge.py
│   └── model_client.py
├── app.py
├── pipeline.py
├── data/
│   └── clips/
└── results.json
```

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
