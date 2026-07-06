import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    from src.model_client import ModelClient
except ImportError:
    try:
        from model_client import ModelClient
    except ImportError:
        # Fallback for importing when run inside other contexts
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_client import ModelClient

def describe_scene(frames: Any, transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the scene description step using the ModelClient.
    
    Args:
        frames (Any): List of image paths (strings) or a list of frame dicts (e.g. [{"frame": path}]).
        transcript (Dict[str, Any]): Dictionary containing:
                                      - 'transcript': str
                                      - 'confidence': float
                                      - 'has_speech': bool

    Returns:
        Dict[str, Any]: Validated structured scene description with 5 required keys.
    """
    # 1. Input validation for frames
    if not frames:
        raise ValueError("Input validation failed: 'frames' list cannot be empty.")

    frame_paths = []
    if isinstance(frames, list):
        for item in frames:
            if isinstance(item, dict) and "frame" in item:
                frame_paths.append(item["frame"])
            elif isinstance(item, str):
                frame_paths.append(item)
            else:
                logging.warning(f"Skipping invalid frame item: {item}")
    else:
        raise TypeError("Input validation failed: 'frames' must be a list.")

    if not frame_paths:
        raise ValueError("Input validation failed: No valid frame image paths found in 'frames' list.")

    # 2. Input validation for transcript
    if not isinstance(transcript, dict):
        raise TypeError("Input validation failed: 'transcript' must be a dict.")

    has_speech = transcript.get("has_speech", False)
    transcript_text = transcript.get("transcript", "").strip()

    # 3. Audio context logic to prevent dialogue hallucination
    if not has_speech:
        model_transcript = "No speech detected, audio may contain music/ambient sound only"
    else:
        model_transcript = transcript_text if transcript_text else "No speech detected, audio may contain music/ambient sound only"

    # 4. Invoke ModelClient
    client = ModelClient()
    scene_desc = client.describe_scene(frame_paths, model_transcript)

    # 5. Schema validation step
    required_keys = {
        "setting": "unknown",
        "subjects": [],
        "actions": [],
        "notable_details": [],
        "audio_context": "unknown"
    }

    validated_scene = {}
    filled_defaults = []

    for key, default_val in required_keys.items():
        if key not in scene_desc or scene_desc[key] is None:
            validated_scene[key] = default_val
            filled_defaults.append(key)
        else:
            val = scene_desc[key]
            # Ensure type safety (lists are lists, strings are strings)
            if isinstance(default_val, list) and not isinstance(val, list):
                validated_scene[key] = [val] if val else []
                filled_defaults.append(key)
            elif isinstance(default_val, str) and not isinstance(val, str):
                validated_scene[key] = str(val)
                filled_defaults.append(key)
            else:
                validated_scene[key] = val

    if filled_defaults:
        logging.warning(f"Schema validation filled in defaults for missing/invalid keys: {filled_defaults}")

    return validated_scene

if __name__ == "__main__":
    import sys
    import json
    import glob

    if len(sys.argv) < 2:
        print("Usage: python src/scene_understanding.py <clip_id>")
        sys.exit(1)

    clip_id = sys.argv[1]
    clip_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clip_id)

    print(f"Testing scene_understanding for clip_id: {clip_id}")

    # Load frames
    frames_dir = f"data/clips/{clip_id}/frames"
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    if not frame_paths:
        print(f"Error: No frames found in {frames_dir}. Please run extraction first.")
        sys.exit(1)

    # Load transcript if exists
    transcript_path = f"data/clips/{clip_id}/transcript.json"
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r") as f:
                transcript = json.load(f)
            print(f"Loaded transcript from {transcript_path}")
        except Exception as e:
            print(f"Error loading transcript file: {e}")
            transcript = {"transcript": "", "confidence": 0.0, "has_speech": False}
    else:
        print(f"No transcript.json found at {transcript_path}. Using default empty transcript.")
        transcript = {"transcript": "", "confidence": 0.0, "has_speech": False}

    try:
        scene_json = describe_scene(frame_paths, transcript)
        print("\nResulting Scene JSON:")
        print(json.dumps(scene_json, indent=4))
    except Exception as e:
        print(f"Error running describe_scene: {e}")
