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

def validate_scores(scores: Any) -> bool:
    """
    Checks that the scores object is a dictionary containing 'accuracy' and 'tone_fit' 
    keys with integer values in the range 1-5.
    """
    if not isinstance(scores, dict):
        return False
    if "accuracy" not in scores or "tone_fit" not in scores:
        return False
    try:
        accuracy = int(scores["accuracy"])
        tone_fit = int(scores["tone_fit"])
        if 1 <= accuracy <= 5 and 1 <= tone_fit <= 5:
            return True
    except (ValueError, TypeError):
        pass
    return False

def judge_caption(caption: str, scene_json: Dict[str, Any], tone: str) -> Dict[str, int]:
    """
    Evaluates the quality of a generated caption based on accuracy and tone alignment.
    
    Args:
        caption (str): The generated caption string.
        scene_json (Dict[str, Any]): The structured scene data.
        tone (str): The target tone.

    Returns:
        Dict[str, int]: A dictionary containing evaluation scores:
                        - 'accuracy': Score from 1 to 5 indicating how accurately 
                                      the caption reflects the scene.
                        - 'tone_fit': Score from 1 to 5 indicating how well 
                                      the caption matches the desired tone.
                        If evaluation fails, defaults to {"accuracy": 0, "tone_fit": 0}.
    """
    client = ModelClient()
    
    # Attempt 1
    try:
        result = client.judge_caption(caption, scene_json, tone)
        if validate_scores(result):
            return {
                "accuracy": int(result["accuracy"]),
                "tone_fit": int(result["tone_fit"])
            }
        else:
            logging.warning(
                f"Invalid scores received on attempt 1 for tone '{tone}': {result}. "
                f"Values must be integers between 1 and 5."
            )
    except Exception as e:
        logging.warning(f"Failed to judge caption on attempt 1 for tone '{tone}': {e}")

    # Retry once
    logging.info(f"Retrying caption judgment for tone '{tone}'...")
    try:
        result = client.judge_caption(caption, scene_json, tone)
        if validate_scores(result):
            return {
                "accuracy": int(result["accuracy"]),
                "tone_fit": int(result["tone_fit"])
            }
        else:
            logging.warning(
                f"Invalid scores received on retry for tone '{tone}': {result}."
            )
    except Exception as e:
        logging.warning(f"Failed to judge caption on retry for tone '{tone}': {e}")

    # Fallback to distinct error scores
    logging.warning(
        f"Failed to obtain valid judgment scores for tone '{tone}' after retry. "
        f"Defaulting to accuracy=0, tone_fit=0."
    )
    return {"accuracy": 0, "tone_fit": 0}

def judge_all_captions(captions_dict: Dict[str, str], scene_json: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Evaluates generated captions for all tones.

    Args:
        captions_dict (Dict[str, str]): Generated captions mapped by tone.
        scene_json (Dict[str, Any]): Structured details of the scene.

    Returns:
        Dict[str, Dict[str, int]]: Dictionary mapping each tone to its judgment scores.
    """
    scores = {}
    for tone, caption in captions_dict.items():
        scores[tone] = judge_caption(caption, scene_json, tone)
    return scores

if __name__ == "__main__":
    import json

    # Mock scene JSON
    mock_scene = {
        "setting": "A Formula 1 podium celebration under clear skies.",
        "subjects": ["An F1 driver in a red suit", "Cheering crew members"],
        "actions": ["Driver spraying champagne", "Crew members celebrating"],
        "notable_details": ["A gold trophy", "Champagne bottle"],
        "audio_context": "Loud music and crowd cheering."
    }

    # Mock generated captions
    mock_captions = {
        "formal": "The driver celebrates their victory on the podium with the team.",
        "sarcastic": "Look at them celebrating like they actually did all the work.",
        "humorous_tech": "Podium system successfully compiled with 1 champagne bug.",
        "humorous_non_tech": "F1 driver uses champagne spray! It is highly effective!"
    }

    print("=== Testing self_judge.py ===")
    print("Scene JSON:")
    print(json.dumps(mock_scene, indent=2))
    print("\nMock Captions:")
    print(json.dumps(mock_captions, indent=2))

    print("\nJudging all captions (Mock Mode will be auto-triggered)...")
    scores = judge_all_captions(mock_captions, mock_scene)
    
    print("\nResulting Scores:")
    print(json.dumps(scores, indent=4))
