import os
import sys
import json
import glob
import time
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    from src.extraction import extract_frames, extract_transcript
    from src.scene_understanding import describe_scene
    from src.caption_generation import generate_all_captions
    from src.model_client import ModelClient
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from extraction import extract_frames, extract_transcript
    from scene_understanding import describe_scene
    from caption_generation import generate_all_captions
    from model_client import ModelClient

def write_results(output_path, results_list):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_list, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")

def run_batch_evaluation():
    # Reset usage counters
    ModelClient.reset_usage()

    # Determine input path
    tasks_path = "/input/tasks.json"
    if not os.path.exists(tasks_path):
        tasks_path = "input/tasks.json"

    tasks = []
    is_local_scan = False

    if os.path.exists(tasks_path):
        try:
            with open(tasks_path, "r") as f:
                tasks = json.load(f)
            logging.info(f"Loaded {len(tasks)} tasks from {tasks_path}")
        except Exception as e:
            logging.error(f"Failed to read {tasks_path}: {e}")
            is_local_scan = True
    else:
        is_local_scan = True

    if is_local_scan:
        logging.info("Falling back to local data/clips/ scanning...")
        clips_dir = "data/clips"
        extensions = ("*.mp4", "*.mov", "*.avi")
        local_files = []
        for ext in extensions:
            local_files.extend(glob.glob(os.path.join(clips_dir, ext)))
            local_files.extend(glob.glob(os.path.join(clips_dir, ext.upper())))
        local_files = sorted(list(set(local_files)))
        
        for filepath in local_files:
            filename = os.path.basename(filepath)
            task_id = os.path.splitext(filename)[0]
            tasks.append({
                "task_id": task_id,
                "local_path": filepath
            })
        logging.info(f"Found {len(tasks)} local clips to process.")

    # Determine output path
    output_file = "/output/results.json"
    os.makedirs("/output", exist_ok=True)

    # Determine temp folder for downloads
    temp_dir = "/tmp"
    if sys.platform.startswith("win"):
        temp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    results_list = []

    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        local_path = task.get("local_path")
        
        logging.info(f"[{idx}/{len(tasks)}] Processing task_id: {task_id}")
        
        video_path = None
        is_temp_file = False
        
        try:
            # 1. Resolve video path (Download if URL)
            if local_path:
                video_path = local_path
                logging.info(f"Using local video file: {video_path}")
            elif video_url:
                dest_path = os.path.join(temp_dir, f"{task_id}.mp4")
                logging.info(f"Downloading {video_url} to {dest_path}")
                
                response = requests.get(video_url, stream=True, timeout=(10, 110))
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            
                video_path = dest_path
                is_temp_file = True
            else:
                raise ValueError("Task contains neither 'video_url' nor 'local_path'")

            # 2. Extract frames & transcript
            logging.info("Extracting frames & transcript...")
            frames = extract_frames(video_path, fps_sample=1.0, max_frames=8)
            transcript = extract_transcript(video_path)

            # 3. Describe scene
            logging.info("Generating structured scene understanding...")
            scene_json = describe_scene(frames, transcript)

            # 4. Generate all captions
            logging.info("Generating 4-tone captions...")
            captions = generate_all_captions(scene_json)

            # Assemble successful results dict
            task_result = {
                "task_id": task_id,
                "captions": captions
            }
            results_list.append(task_result)
            logging.info(f"SUCCESS for task_id: {task_id}")

        except Exception as e:
            logging.error(f"FAILURE for task_id {task_id}: {e}", exc_info=True)
            task_result = {
                "task_id": task_id,
                "error": str(e),
                "captions": None
            }
            results_list.append(task_result)

        finally:
            # 5. Delete temporary video file immediately after processing
            if is_temp_file and video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logging.info(f"Deleted temporary video file: {video_path}")
                except Exception as del_err:
                    logging.warning(f"Failed to delete temp file {video_path}: {del_err}")

        # 6. Write results incrementally after every clip
        write_results(output_file, results_list)

        # Rate limit pause between clips on free tier (1 second)
        if idx < len(tasks):
            logging.info("Pausing 1 second before next task...")
            time.sleep(1)

    logging.info("Batch evaluation job finished.")
    ModelClient.print_usage_summary()

    # Exit with 0 status code
    sys.exit(0)

if __name__ == "__main__":
    run_batch_evaluation()
    sys.exit(0)