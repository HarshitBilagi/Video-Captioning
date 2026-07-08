import os
import base64
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in .env")

# CHANGE THIS to a real extracted frame from one of your clips
FRAME_PATH = "data/clips/Clip2/frames/frame_5.000.jpg"

with open(FRAME_PATH, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

print(f"Image encoded, base64 size: {len(b64_image)} chars (~{len(b64_image)/1024:.1f} KB)")

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
    "model": "accounts/fireworks/models/minimax-m3",
    "max_tokens": 500,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what is happening in this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        }
    ]
}
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

print("Sending request...")
response = requests.post(url, headers=headers, data=json.dumps(payload))
print(f"Status code: {response.status_code}")
print(json.dumps(response.json(), indent=2))