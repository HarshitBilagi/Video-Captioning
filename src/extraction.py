from typing import List, Tuple, Any

def extract_frames(video_path: str) -> List[Tuple[Any, float]]:
    """
    Extracts keyframes from the video along with their corresponding timestamps.

    Args:
        video_path (str): The local path to the video file.

    Returns:
        List[Tuple[Any, float]]: A list of tuples containing the image frame data (e.g., numpy array) 
                                 and the timestamp of the frame in seconds.
    """
    return []

def extract_transcript(video_path: str) -> Tuple[str, float]:
    """
    Extracts the audio transcript from the video using a transcription model.

    Args:
        video_path (str): The local path to the video file.

    Returns:
        Tuple[str, float]: A tuple containing the transcribed text and the average confidence score (0.0 to 1.0).
    """
    return "", 0.0
