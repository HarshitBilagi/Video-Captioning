import json
import os
from typing import Dict, Any
from src.extraction import extract_frames, extract_transcript
from src.scene_understanding import describe_scene
from src.caption_generation import generate_caption
from src.self_judge import judge_caption
from src.model_client import ModelClient

def run_pipeline(video_path: str, tone: str) -> Dict[str, Any]:
    """
    Runs the complete video captioning pipeline on the specified video.

    Args:
        video_path (str): Path to the video file.
        tone (str): Target caption tone.

    Returns:
        Dict[str, Any]: The pipeline execution results including extracted details, caption, and judgment.
    """
    # 1. Initialize client
    client = ModelClient()
    
    # 2. Extract frames & transcript
    frames = extract_frames(video_path)
    transcript, confidence = extract_transcript(video_path)
    
    # 3. Scene understanding
    scene_json = describe_scene(frames, transcript)
    
    # 4. Caption generation
    caption = generate_caption(scene_json, tone)
    
    # 5. Self-judgment
    evaluation = judge_caption(caption, scene_json, tone)
    
    # Prepare result structure
    result = {
        "video_path": video_path,
        "tone": tone,
        "transcript": transcript,
        "scene_description": scene_json,
        "caption": caption,
        "evaluation": evaluation
    }
    
    # 6. Save or append to results.json
    results_file = "results.json"
    results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = []
            
    results.append(result)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return result

if __name__ == "__main__":
    # Test or dry run execution placeholder
    print("Video Captioning Pipeline initialized.")
