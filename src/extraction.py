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
    # pyrefly: ignore [missing-import]
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

class VideoProcessingError(Exception):
    """Custom exception raised during video processing error scenarios."""
    pass

def _get_config_value(key: str, default: Any) -> Any:
    # 1. Try env var first
    val = os.getenv(key)
    if val is not None:
        return val
    # 2. Try models.yaml
    try:
        import yaml
        config_path = "config/models.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get(key.lower(), default)
    except Exception:
        pass
    return default

def extract_frames(video_url: str, max_frames: int = 8) -> List[str]:
    """
    Extracts keyframes from the video URL directly using a grab()/retrieve() pattern.
    Applies green-frame validation, resizing to max 512px width, and outputs a list of base64 JPEG strings.
    """
    # 1. Load config parameters
    min_seconds = float(_get_config_value("MIN_VIDEO_SECONDS", 2))
    max_seconds = float(_get_config_value("MAX_VIDEO_SECONDS", 300))
    frames_per_second = float(_get_config_value("FRAMES_PER_SECOND", 1.0))
    jpeg_quality = int(_get_config_value("FRAME_JPEG_QUALITY", 70))

    if max_frames <= 0:
        logging.warning("max_frames is less than or equal to 0. Returning empty list.")
        return []

    cap = None
    try:
        logging.info(f"Opening video stream directly from: {video_url}")
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video source: {video_url}")

        # 2. Determine video duration and fps from the stream
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or fps <= 0:
            logging.warning("Unreliable stream stats. Attempting to deduce details...")
            duration = 0.0
        else:
            duration = total_frames / fps

        # 8. Add duration bounds checking before extraction begins
        if duration > 0.0 and (duration < min_seconds or duration > max_seconds):
            raise VideoProcessingError(
                f"Video duration {duration:.2f}s is outside configured bounds [{min_seconds}, {max_seconds}]."
            )

        # 3. Extract frames using grab()/retrieve() pattern
        frames_to_skip = int(fps / frames_per_second)
        if frames_to_skip <= 0:
            frames_to_skip = 1

        extracted_count = 0
        base64_frames = []
        current_frame_idx = 0

        while extracted_count < max_frames:
            ret = cap.grab()  # advance without decoding
            if not ret:
                break
            
            if current_frame_idx % frames_to_skip == 0:
                ret, frame = cap.retrieve()  # decode only needed frames
                if not ret or frame is None:
                    current_frame_idx += 1
                    continue

                # 4. Green-frame sanity check
                try:
                    mean_b, mean_g, mean_r = frame.mean(axis=(0, 1))
                    if mean_g > 200 and mean_r < 30 and mean_b < 30:
                        logging.warning(f"Sanity check failed: Skipping green/corrupted frame at index {current_frame_idx}.")
                        current_frame_idx += 1
                        continue
                except Exception as check_err:
                    logging.warning(f"Sanity check error, keeping frame: {check_err}")

                # 5. Resize every valid frame to max 512px width, preserving aspect ratio
                height, width = frame.shape[:2]
                if width > 512:
                    scale = 512.0 / width
                    frame = cv2.resize(frame, (512, int(height * scale)), interpolation=cv2.INTER_AREA)

                # 6. Encode each frame as base64 JPEG at quality 70 (or jpeg_quality)
                import base64
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if success:
                    base64_str = base64.b64encode(buffer).decode("utf-8")
                    base64_frames.append(base64_str)
                    extracted_count += 1

            current_frame_idx += 1

        # Fallback to first frame read if no frames were retrieved
        if not base64_frames:
            logging.info("Attempting first-frame fallback extraction...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                if width > 512:
                    scale = 512.0 / width
                    frame = cv2.resize(frame, (512, int(height * scale)), interpolation=cv2.INTER_AREA)
                import base64
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if success:
                    base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        return base64_frames

    except Exception as e:
        if isinstance(e, VideoProcessingError):
            raise e
        raise VideoProcessingError(f"Error during video frame extraction stream: {e}")

    finally:
        # 9. Release the VideoCapture object in a finally block
        if cap is not None:
            cap.release()

def get_video_duration(video_path: str) -> float:
    """Reads the video duration in seconds using OpenCV."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and total_frames > 0:
                return total_frames / fps
    except Exception as e:
        logging.error(f"Error reading video duration: {e}")
    finally:
        if cap is not None:
            cap.release()
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
        # Run ffmpeg silently with a hard 120 second timeout to prevent hangs
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=120)
        if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
            return temp_wav
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "does not contain any stream" in stderr or "Output file does not contain any stream" in stderr:
            logging.info(f"Video {video_path} does not contain any audio stream.")
        else:
            logging.error(f"ffmpeg failed to extract audio: {stderr}")
    except subprocess.TimeoutExpired:
        logging.error(f"ffmpeg extraction timed out after 120 seconds on {video_path}")
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

async def transcribe_audio(audio_path: str, duration: float) -> Dict[str, Any]:
    """Transcribes a WAV file using Groq's hosted Whisper API."""
    import httpx
    
    # 1. Load API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        if not os.getenv("FIREWORKS_API_KEY"):
            logging.info("GROQ_API_KEY is missing but running in MOCK MODE. Returning mock transcript.")
            return {
                "transcript": "[MOCK] This is a mock audio transcription for testing.",
                "confidence": 0.9,
                "has_speech": True
            }
        raise ValueError("GROQ_API_KEY is missing. Please set it in your environment or .env file.")

    # 2. Load model name
    model_name = "whisper-large-v3"
    try:
        import yaml
        config_path = "config/models.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                model_name = cfg.get("groq_whisper_model", "whisper-large-v3")
    except Exception:
        pass

    logging.info(f"Sending audio transcription request to Groq API using model '{model_name}'...")
    
    transcript = ""
    confidence = 0.0
    has_speech = False

    try:
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/wav")
            }
            data = {
                "model": model_name,
                "response_format": "verbose_json"
            }
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files
                )
                response.raise_for_status()
                result = response.json()

        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])

        if segments:
            confidences = []
            for s in segments:
                # Calculate confidence probability from avg_logprob
                logprob = s.get("avg_logprob")
                if logprob is not None:
                    prob = math.exp(max(-10.0, min(0.0, float(logprob))))
                    confidences.append(prob)
            
            if confidences:
                confidence = sum(confidences) / len(confidences)
            else:
                confidence = get_heuristic_confidence(transcript, duration)

            # Get average no_speech_prob
            no_speech_probs = [float(s.get("no_speech_prob", 0.0)) for s in segments if s.get("no_speech_prob") is not None]
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
            # Check transcript length as fallback if no segments returned
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

    except Exception as e:
        logging.error(f"Groq transcription API request failed: {e}")
        return {
            "transcript": "",
            "confidence": 0.0,
            "has_speech": False
        }

    return {
        "transcript": transcript,
        "confidence": float(confidence),
        "has_speech": has_speech
    }

async def extract_transcript(video_path: str) -> Dict[str, Any]:
    """
    Extracts the audio transcript from the video using Groq's hosted Whisper API.

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
        # 2. Run Groq Whisper on the audio file
        result = await transcribe_audio(temp_wav, duration)
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
