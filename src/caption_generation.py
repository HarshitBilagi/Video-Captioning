import re
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    from src.model_client import ModelClient
except ImportError:
    try:
        from model_client import ModelClient
    except ImportError:
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_client import ModelClient

VALID_TONES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

def clean_and_truncate_caption(caption: str, max_len: int = 280) -> str:
    """
    Cleans the caption by stripping surrounding quotes, collapsing extra whitespace,
    and truncating at a word boundary if it exceeds max_len.
    """
    # 1. Strip surrounding quotes and whitespace
    cleaned = caption.strip().strip(" \t\n\r\"'")
    
    # 2. Collapse extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 3. Enforce max length by truncating at a word boundary
    if len(cleaned) > max_len:
        limit = max_len - 3  # Leave room for ellipsis "..."
        truncated = cleaned[:limit]
        
        # Find the last space to avoid cutting a word in half
        last_space = truncated.rfind(' ')
        if last_space != -1:
            truncated = truncated[:last_space]
            
        cleaned_truncated = truncated.rstrip(".,?!:;-") + "..."
        logging.warning(
            f"Caption length of {len(cleaned)} chars exceeded max_len ({max_len}). "
            f"Truncated to: '{cleaned_truncated}'"
        )
        return cleaned_truncated
        
    return cleaned

def generate_caption(scene_json: Dict[str, Any], tone: str) -> str:
    """
    Generates a caption for the video based on the structured scene details and a specified tone.

    Args:
        scene_json (Dict[str, Any]): Structured details about the video scene.
        tone (str): The desired tone for the caption. Valid options: 
                    'formal', 'sarcastic', 'humorous_tech', 'humorous_non_tech'.

    Returns:
        str: The generated, post-processed caption.
    """
    # 1. Validate tone
    if tone not in VALID_TONES:
        raise ValueError(
            f"Invalid tone: '{tone}'. Valid options are: {VALID_TONES}"
        )
        
    # 2. Call ModelClient
    client = ModelClient()
    raw_caption = client.generate_caption(scene_json, tone)
    
    # 3. Apply post-processing
    return clean_and_truncate_caption(raw_caption)

def generate_all_captions(scene_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates captions for all 4 supported tones.

    Args:
        scene_json (Dict[str, Any]): Structured details of the scene.

    Returns:
        Dict[str, str]: Dictionary mapping each tone to its generated caption.
    """
    captions = {}
    for tone in VALID_TONES:
        captions[tone] = generate_caption(scene_json, tone)
    return captions

if __name__ == "__main__":
    import sys
    import json
    import glob
    import os

    if len(sys.argv) < 2:
        print("Usage: python src/caption_generation.py <clip_id>")
        sys.exit(1)

    clip_id = sys.argv[1]
    clip_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clip_id)

    print(f"Testing caption_generation for clip_id: {clip_id}")

    # Resolve scene JSON
    scene_json = None
    
    # Try loading from scene_description.json
    scene_desc_path = f"data/clips/{clip_id}/scene_description.json"
    if os.path.exists(scene_desc_path):
        try:
            with open(scene_desc_path, "r") as f:
                scene_json = json.load(f)
            print(f"Loaded scene JSON from {scene_desc_path}")
        except Exception as e:
            print(f"Error loading {scene_desc_path}: {e}")

    # Try generating using scene_understanding if frames are available
    if scene_json is None:
        frames_dir = f"data/clips/{clip_id}/frames"
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if frame_paths:
            try:
                from scene_understanding import describe_scene
                print("Generating scene description from extracted frames...")
                scene_json = describe_scene(frame_paths, {"transcript": "", "confidence": 0.0, "has_speech": False})
            except Exception as e:
                print(f"Could not describe scene: {e}")

    # Fallback to realistic mock scene JSON to run fully in mock mode
    if scene_json is None:
        print("No scene file or frames found. Using default mock scene details.")
        scene_json = {
            "setting": "A brightly lit Formula 1 podium area and pitlane celebration scene under daylight.",
            "subjects": [
                "An F1 driver wearing a racing suit", 
                "Team members in team uniforms"
            ],
            "actions": [
                "F1 driver celebrating on the podium", 
                "Team members cheering and waving flags"
            ],
            "notable_details": [
                "Metallic trophy in hand", 
                "Sponsor branding on podium board"
            ],
            "audio_context": "Crowd cheering and engines humming."
        }

    print("\nScene context used for captioning:")
    print(json.dumps(scene_json, indent=2))

    try:
        print("\nGenerating captions for all tones...")
        captions = generate_all_captions(scene_json)
        print("\nGenerated Captions:")
        for tone, caption in captions.items():
            print(f"  {tone.upper()}: {caption}")
    except Exception as e:
        print(f"Error generating captions: {e}")
