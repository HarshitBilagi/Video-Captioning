import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    raise ValueError("FIREWORKS_API_KEY not found in .env")

URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def call(messages, max_tokens=500, label=""):
    payload = {
        "model": "accounts/fireworks/models/minimax-m3",
        "max_tokens": max_tokens,
        "messages": messages
    }
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"{'='*50}")
    response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
    print(f"Status: {response.status_code}")
    data = response.json()
    if "choices" not in data:
        print(f"Unexpected response: {json.dumps(data, indent=2)}")
        return "[API ERROR]"
    message = data["choices"][0]["message"]
    print(f"Full message object: {json.dumps(message, indent=2)}")  # temporary debug line
    content = message.get("content") or message.get("reasoning_content") or "[NO CONTENT]"
    usage = data["usage"]
    print(f"Response:\n{content}")
    print(f"Tokens — prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']}, total: {usage['total_tokens']}")
    return content

# ── TEST 1: Text-only caption generation ──────────────────────────────────────
sample_scene = {
    "setting": "An outdoor park pathway surrounded by trees",
    "subjects": ["Two people on a paved road"],
    "actions": ["One person standing upright", "Another person bent forward dramatically"],
    "notable_details": ["Text overlay reading 'Recreate this whenever you are out next time'", "Tilted diagonal image format"],
    "audio_context": "No speech detected. Audio may contain music or ambient sound."
}

for tone, instruction in [
    ("formal",           "Write a formal, objective, third-person caption for this scene. Be precise and professional. No humor or judgment."),
    ("sarcastic",        "Write a dry, deadpan sarcastic caption for this scene. Say the opposite of what you mean or exaggerate the mundane as remarkable. No exclamation marks. No emojis."),
    ("humorous_tech",    "Write a humorous caption framing this scene entirely using programming or tech metaphors. The humor must come from the tech framing itself, not just added jargon."),
    ("humorous_non_tech","Write a funny, relatable, observational caption for this scene using everyday language. No tech jargon. Situational humor only."),
]:
    call([
        {"role": "system", "content": f"{instruction}"},
        {"role": "user",   "content": f"Scene details:\n{json.dumps(sample_scene, indent=2)}\n\nGenerate one caption only. No preamble, no quotes around it."}
    ], max_tokens=150, label=f"Caption — {tone}")

# ── TEST 2: Structured JSON scene output ──────────────────────────────────────
call([
    {
        "role": "system",
        "content": (
            "You are a scene analysis assistant. "
            "Given a description of a video scene, return ONLY a valid JSON object with exactly these keys: "
            "setting (string), subjects (array of strings), actions (array of strings), "
            "notable_details (array of strings), audio_context (string). "
            "No markdown fences. No explanation. No extra keys. Return raw JSON only."
        )
    },
    {
        "role": "user",
        "content": (
            "Analyze this scene and return structured JSON:\n"
            "Two people are on an outdoor paved pathway. One stands upright in a light blue hoodie "
            "looking at the other. The second person is bent dramatically forward at the waist toward "
            "the ground wearing denim shorts. There is a green mesh barrier in the background and lush "
            "trees. A text overlay reads 'Recreate this whenever you are out next time'. "
            "No speech in audio, background music likely present."
        )
    }
], max_tokens=400, label="Structured JSON scene output")