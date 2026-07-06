import os
import cv2
import logging
import math
import subprocess
import shutil
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure ffmpeg is in PATH for Whisper compatibility on Windows/macOS/Linux
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    
    # On Windows, we need 'ffmpeg.exe'. If it doesn't exist, make a copy of the versioned executable.
    target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    target_ffmpeg = os.path.join(ffmpeg_dir, target_name)
    
    if not os.path.exists(target_ffmpeg):
        try:
            shutil.copy2(ffmpeg_exe, target_ffmpeg)
            logging.info(f"Created copy of ffmpeg executable at: {target_ffmpeg}")
        except Exception as e:
            logging.warning(f"Could not create a copy of ffmpeg at {target_ffmpeg}: {e}")
            
    # Add imageio_ffmpeg binaries folder to PATH
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    logging.warning("imageio_ffmpeg not installed. Whisper subprocess might fail if system ffmpeg is missing.")

def compute_frame_diff(frame1: Any, frame2: Any) -> float:
    """
    Computes a pixel-based difference score between two frames.
    Resizes frames to a small scale and uses absolute difference.
    """
    if frame1 is None or frame2 is None:
        return 0.0
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray1_resized = cv2.resize(gray1, (100, 100))
        gray2_resized = cv2.resize(gray2, (100, 100))
        diff = cv2.absdiff(gray1_resized, gray2_resized)
        return float(diff.mean())
    except Exception as e:
        logging.error(f"Error computing frame diff: {e}")
        return 0.0

def extract_frames(video_path: str, fps_sample: float = 1.0, max_frames: int = 8) -> List[Dict[str, Any]]:
    """
    Extracts keyframes from the video using OpenCV.
    
    Requirements:
    - Sample 1 frame per second by default (configurable via fps_sample)
    - Detect scene changes using frame-diff and prioritize keeping frames at transitions over static frames
    - Return a list of dicts: [{"timestamp": float, "frame": path_to_saved_jpg}]
    - Save extracted frames to a temp directory (data/clips/{clip_id}/frames/) as jpgs
    - Handle short clips gracefully (extract at least 1 frame)
    - Add basic error handling for corrupted/unreadable videos
    """
    if max_frames <= 0:
        logging.warning("max_frames is less than or equal to 0. Returning empty list.")
        return []

    if not os.path.exists(video_path):
        logging.error(f"Video file does not exist: {video_path}")
        return []

    # Create temporary directory for clips
    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    clip_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clip_id)
    output_dir = f"data/clips/{clip_id}/frames"
    os.makedirs(output_dir, exist_ok=True)

    candidates: List[Tuple[Any, float]] = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            logging.warning(f"Unreliable FPS ({fps}) detected. Falling back to 30.0 FPS.")
            fps = 30.0

        step_seconds = 1.0 / fps_sample
        step_frames = max(1, int(round(fps * step_seconds)))

        seek_supported = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if seek_supported and total_frames > 0:
            current_frame = 0
            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = current_frame / fps
                candidates.append((frame, timestamp))
                current_frame += step_frames
        else:
            # Fallback sequential read
            current_frame = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame % step_frames == 0:
                    timestamp = current_frame / fps
                    candidates.append((frame, timestamp))
                current_frame += 1

        cap.release()

    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")
        return []

    # Handle short clips gracefully: if we extracted nothing but there is a frame, read at least frame 0
    if not candidates:
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                candidates.append((frame, 0.0))
            cap.release()
        except Exception as e:
            logging.error(f"Failed to extract fallback first frame: {e}")

    if not candidates:
        logging.warning(f"No frames could be extracted from {video_path}")
        return []

    # Prioritize scene transition frames if candidate count exceeds max_frames
    selected_candidates: List[Tuple[Any, float]] = []
    if len(candidates) <= max_frames:
        selected_candidates = candidates
    else:
        # Compute differences between consecutive candidate frames
        diffs = [0.0]  # The first frame has no previous frame to compare
        for i in range(1, len(candidates)):
            diff = compute_frame_diff(candidates[i][0], candidates[i-1][0])
            diffs.append(diff)

        # We always keep the first frame (index 0).
        # We select the top (max_frames - 1) frames with the highest transition score from the rest.
        indexed_diffs = [(i, diffs[i]) for i in range(1, len(candidates))]
        indexed_diffs.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in indexed_diffs[:max_frames - 1]]
        selected_indices = [0] + top_indices
        selected_indices.sort()  # Restore chronological order

        selected_candidates = [candidates[i] for i in selected_indices]

    # Save selected frames to temp directory and prepare the result list
    results = []
    for frame, timestamp in selected_candidates:
        frame_filename = os.path.join(output_dir, f"frame_{timestamp:.3f}.jpg")
        normalized_path = os.path.normpath(frame_filename).replace("\\", "/")
        try:
            success = cv2.imwrite(normalized_path, frame)
            if success:
                results.append({
                    "timestamp": float(timestamp),
                    "frame": normalized_path
                })
            else:
                logging.error(f"Failed to write frame at timestamp {timestamp} to {normalized_path}")
        except Exception as e:
            logging.error(f"Exception saving frame {normalized_path}: {e}")

    return results

def get_video_duration(video_path: str) -> float:
    """Reads the video duration in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and total_frames > 0:
                return total_frames / fps
    except Exception as e:
        logging.error(f"Error reading video duration: {e}")
    return 0.0

def extract_audio_from_video(video_path: str) -> str:
    """Extracts the audio track from the video file to a temp WAV file using ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"

    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    clip_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clip_id)
    temp_dir = os.path.join("data", "clips", "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    temp_wav = os.path.join(temp_dir, f"{clip_id}_temp.wav")

    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_wav
    ]

    try:
        # Run ffmpeg silently
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
            return temp_wav
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "does not contain any stream" in stderr or "Output file does not contain any stream" in stderr:
            logging.info(f"Video {video_path} does not contain any audio stream.")
        else:
            logging.error(f"ffmpeg failed to extract audio: {stderr}")
    except Exception as e:
        logging.error(f"Unexpected error extracting audio from video: {e}")

    return None

def get_heuristic_confidence(transcript: str, duration: float) -> float:
    """Computes a heuristic confidence score based on transcript length vs audio duration."""
    if not transcript or duration <= 0:
        return 0.0
    words = len(transcript.split())
    if words == 0:
        return 0.0
    words_per_second = words / duration
    if 0.5 <= words_per_second <= 4.0:
        return 0.85
    elif words_per_second > 4.0:
        return max(0.2, 0.85 - (words_per_second - 4.0) * 0.1)
    else:
        return max(0.2, 0.85 - (0.5 - words_per_second) * 0.5)

def _get_transcription_model_name() -> str:
    """Loads the transcription model size from configuration."""
    try:
        import yaml
        config_path = "config/models.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                model = cfg.get("transcription_model", "tiny")
                # Map placeholders or return
                if model == "whisper-v3":
                    return "tiny"  # Use tiny locally for fast execution and resources
                return model
    except Exception:
        pass
    return "tiny"

def transcribe_audio(audio_path: str, model_size: str, duration: float) -> Dict[str, Any]:
    """Transcribes a WAV file using faster-whisper or openai-whisper."""
    transcript = ""
    confidence = 0.0
    has_speech = False

    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel
        logging.info(f"Using faster-whisper with model size '{model_size}'")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logging.warning(f"Failed to load faster-whisper model '{model_size}': {e}. Falling back to 'tiny'.")
            model = WhisperModel("tiny", device=device, compute_type=compute_type)

        segments, info = model.transcribe(audio_path, beam_size=5)
        segments_list = list(segments)

        if segments_list:
            texts = [s.text.strip() for s in segments_list if s.text.strip()]
            transcript = " ".join(texts)

            confidences = []
            for s in segments_list:
                prob = math.exp(max(-10.0, min(0.0, s.avg_logprob))) if s.avg_logprob else 0.0
                confidences.append(prob)

            if confidences:
                confidence = sum(confidences) / len(confidences)
            else:
                confidence = get_heuristic_confidence(transcript, duration)

            no_speech_probs = [s.no_speech_prob for s in segments_list if s.no_speech_prob is not None]
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0

            # Count alphanumeric characters to filter out punctuation hallucinations (e.g. "!!!")
            alnum_count = sum(1 for c in transcript if c.isalnum())

            if alnum_count >= 2 and avg_no_speech < 0.8 and confidence >= 0.1:
                has_speech = True
            else:
                transcript = ""
                has_speech = False
                confidence = 0.0

        return {
            "transcript": transcript,
            "confidence": float(confidence),
            "has_speech": has_speech
        }
    except ImportError:
        logging.info("faster-whisper not available. Trying openai-whisper...")

    # Fallback to openai-whisper
    try:
        import whisper
        logging.info(f"Using openai-whisper with model size '{model_size}'")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            actual_model = "large-v3" if model_size == "whisper-v3" else model_size
            model = whisper.load_model(actual_model, device=device)
        except Exception as e:
            logging.warning(f"Failed to load openai-whisper model '{model_size}': {e}. Falling back to 'tiny'.")
            model = whisper.load_model("tiny", device=device)

        result = model.transcribe(audio_path)
        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])

        if segments:
            confidences = []
            for s in segments:
                logprob = s.get("avg_logprob", 0.0)
                prob = math.exp(max(-10.0, min(0.0, logprob))) if logprob else 0.0
                confidences.append(prob)

            if confidences:
                confidence = sum(confidences) / len(confidences)
            else:
                confidence = get_heuristic_confidence(transcript, duration)

            no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments]
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0

            # Count alphanumeric characters to filter out punctuation hallucinations
            alnum_count = sum(1 for c in transcript if c.isalnum())

            if alnum_count >= 2 and avg_no_speech < 0.8 and confidence >= 0.1:
                has_speech = True
            else:
                transcript = ""
                has_speech = False
                confidence = 0.0
        else:
            alnum_count = sum(1 for c in transcript if c.isalnum())
            if alnum_count >= 2:
                confidence = get_heuristic_confidence(transcript, duration)
                if confidence >= 0.1:
                    has_speech = True
                else:
                    transcript = ""
                    has_speech = False
                    confidence = 0.0
            else:
                transcript = ""
                has_speech = False
                confidence = 0.0

        return {
            "transcript": transcript,
            "confidence": float(confidence),
            "has_speech": has_speech
        }
    except ImportError:
        logging.error("Neither faster-whisper nor openai-whisper is installed.")
        return {
            "transcript": "",
            "confidence": 0.0,
            "has_speech": False
        }

def extract_transcript(video_path: str) -> Dict[str, Any]:
    """
    Extracts the audio transcript from the video using a transcription model.

    Args:
        video_path (str): The local path to the video file.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        - 'transcript': The transcribed text string.
                        - 'confidence': Average confidence score (0.0 to 1.0).
                        - 'has_speech': Boolean indicating if speech was detected.
    """
    # 1. Extract audio to a temp .wav file
    temp_wav = extract_audio_from_video(video_path)
    if not temp_wav:
        logging.info(f"No audio extracted for {video_path}. Assuming no speech.")
        return {
            "transcript": "",
            "confidence": 0.0,
            "has_speech": False
        }

    duration = get_video_duration(video_path)

    try:
        model_size = _get_transcription_model_name()
        # 2. Run Whisper on the audio file
        result = transcribe_audio(temp_wav, model_size, duration)
        return result
    except Exception as e:
        logging.error(f"Error during transcript extraction: {e}")
        return {
            "transcript": "",
            "confidence": 0.0,
            "has_speech": False
        }
    finally:
        # 3. Clean up temp audio file
        try:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                logging.info(f"Cleaned up temporary audio file: {temp_wav}")
        except Exception as e:
            logging.error(f"Error removing temporary audio file {temp_wav}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/extraction.py <video_path>")
        sys.exit(1)

    video = sys.argv[1]
    print(f"Testing extract_frames on: {video}")
    frames_results = extract_frames(video, fps_sample=1.0, max_frames=8)
    print(f"Extracted {len(frames_results)} frames:")
    for r in frames_results:
        print(f"  Timestamp: {r['timestamp']:.3f}s -> Saved at: {r['frame']}")

    print("\n" + "="*40)
    print(f"Testing extract_transcript on: {video}")
    transcript_result = extract_transcript(video)
    print("Transcript Results:")
    print(f"  Has Speech: {transcript_result['has_speech']}")
    print(f"  Confidence: {transcript_result['confidence']:.4f}")
    print(f"  Transcript: '{transcript_result['transcript']}'")
