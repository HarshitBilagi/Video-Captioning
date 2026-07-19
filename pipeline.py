import os
import sys
import json
import glob
import time
import logging
import asyncio
import httpx

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
        # Write to a temp file first, then replace to guarantee atomic write
        temp_output_path = output_path + ".tmp"
        with open(temp_output_path, "w") as f:
            json.dump(results_list, f, indent=2)
        os.replace(temp_output_path, output_path)
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")

async def process_task(
    idx: int,
    task: dict,
    total_tasks: int,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    start_time: float,
    max_runtime: float,
    results_list: list,
    output_file: str,
    client: ModelClient
):
    async with semaphore:
        # Check elapsed time budget before starting the actual task processing
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            logging.warning(
                f"Approaching time budget ({elapsed_time:.1f}s elapsed), skipping task {task.get('task_id')}."
            )
            # Make sure we write results before returning
            async with file_lock:
                write_results(output_file, results_list)
            return

        task_id = task.get("task_id")
        video_url = task.get("video_url")
        local_path = task.get("local_path")

        logging.info(f"[{idx}/{total_tasks}] Processing task_id: {task_id}")
        video_target = video_url if video_url else local_path
        if not video_target:
            logging.error(f"Task {task_id} contains neither 'video_url' nor 'local_path'")
            task_result = {
                "task_id": task_id,
                "error": "Task contains neither 'video_url' nor 'local_path'",
                "answer": None
            }
            results_list.append(task_result)
            async with file_lock:
                write_results(output_file, results_list)
            return

        try:
            # 2. Extract frames (in thread pool to prevent blocking loop) & transcript (async Groq call)
            logging.info(f"[{idx}/{total_tasks}] Extracting frames & transcript for task_id: {task_id}...")
            frames_task = asyncio.to_thread(extract_frames, video_target)
            transcript_task = extract_transcript(video_target)
            
            frames, transcript = await asyncio.gather(frames_task, transcript_task)

            # 3. Describe scene
            logging.info(f"[{idx}/{total_tasks}] Generating structured scene understanding for task_id: {task_id}...")
            scene_json = await describe_scene(frames, transcript)

            # 4. Generate all captions
            logging.info(f"[{idx}/{total_tasks}] Generating 4-tone captions for task_id: {task_id}...")
            captions = await generate_all_captions(scene_json)

            # 4.5. Self-evaluation and weak-only regeneration
            logging.info(f"[{idx}/{total_tasks}] Performing structured self-evaluation on captions for task_id: {task_id}...")
            try:
                scores = await client.evaluate_all_captions(captions, scene_json)
                logging.info(f"[{idx}/{total_tasks}] Self-evaluation scores for task_id {task_id}: {json.dumps(scores)}")
                
                # Check for weak dimensions (score < 0.6)
                for tone in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]:
                    tone_scores = scores.get(tone, {})
                    accuracy = tone_scores.get("accuracy", 1.0)
                    style_match = tone_scores.get("style_match", 1.0)
                    
                    if accuracy < 0.6 or style_match < 0.6:
                        logging.info(f"[{idx}/{total_tasks}] Weak caption detected for tone '{tone}' in task_id {task_id} (accuracy={accuracy}, style_match={style_match}). Regenerating...")
                        
                        # Build stronger prompt emphasizing the weak dimension
                        from src.model_client import TONE_PROMPTS
                        base_prompt = TONE_PROMPTS.get(tone, "")
                        extra_instruction = ""
                        
                        if accuracy < 0.6 and style_match < 0.6:
                            extra_instruction = " Crucially, make sure you are extremely specific about the exact visual details of the scene AND emphasize the requested tone strongly."
                        elif accuracy < 0.6:
                            extra_instruction = " Crucially, make sure you are extremely specific about the exact visual details of the scene (what is literally visible)."
                        elif style_match < 0.6:
                            if tone == "sarcastic":
                                extra_instruction = " Crucially, make the tone much more distinctly sarcastic, dry, deadpan, and ironic."
                            elif tone == "humorous_tech":
                                extra_instruction = " Crucially, make the tech or programming metaphor punchline much more prominent and distinctly funny."
                            elif tone == "humorous_non_tech":
                                extra_instruction = " Crucially, make the observational everyday humor much more prominent and distinctly funny."
                            else:
                                extra_instruction = f" Crucially, make the tone much more distinctly {tone}."
                        
                        stronger_prompt = base_prompt + extra_instruction
                        
                        # Regenerate caption (only 1 retry capped)
                        try:
                            from src.caption_generation import clean_caption
                            new_raw_caption = await client.generate_caption(scene_json, tone, tone_prompt=stronger_prompt)
                            new_caption = clean_caption(new_raw_caption)
                            
                            logging.info(f"[{idx}/{total_tasks}] Regenerated caption for '{tone}' in task_id {task_id}: {new_caption}")
                            captions[tone] = new_caption
                        except Exception as regen_err:
                            logging.error(f"[{idx}/{total_tasks}] Failed to regenerate caption for tone '{tone}' in task_id {task_id}: {regen_err}")
                            
            except Exception as eval_err:
                logging.error(f"[{idx}/{total_tasks}] Self-evaluation phase failed for task_id {task_id}: {eval_err}. Proceeding with original captions.")

            # Assemble successful results dict matching exact schema
            task_result = {
                "task_id": task_id,
                "answer": {
                    "captions": captions
                }
            }
            results_list.append(task_result)
            logging.info(f"[{idx}/{total_tasks}] SUCCESS for task_id: {task_id}")

        except Exception as e:
            logging.error(f"[{idx}/{total_tasks}] FAILURE for task_id {task_id}: {e}", exc_info=True)
            task_result = {
                "task_id": task_id,
                "error": str(e),
                "answer": None
            }
            results_list.append(task_result)

        # Write results incrementally after every clip completes
        async with file_lock:
            write_results(output_file, results_list)

        # Log remaining time budget after each clip completes
        elapsed = time.time() - start_time
        remaining = max_runtime - elapsed
        logging.info(
            f"Clip {idx}/{total_tasks} done. Elapsed: {elapsed:.1f}s. Budget remaining: {remaining:.1f}s"
        )

async def run_batch_evaluation():
    start_time = time.time()
    MAX_RUNTIME_SECONDS = 540

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
    if sys.platform.startswith("win"):
        output_file = os.path.join(os.getcwd(), "output", "results.json")
    else:
        output_file = "/output/results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results_list = []
    
    # Initialize Concurrency Tools
    semaphore = asyncio.Semaphore(3)
    file_lock = asyncio.Lock()
    client = ModelClient()

    # Build and launch concurrent tasks with staggered starts (1 second gap)
    active_tasks = []
    for idx, task in enumerate(tasks, 1):
        # Check elapsed time budget before launching
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RUNTIME_SECONDS:
            logging.warning(f"Approaching time budget ({elapsed_time:.1f}s elapsed), stopping task dispatch.")
            break
            
        task_coroutine = process_task(
            idx=idx,
            task=task,
            total_tasks=len(tasks),
            semaphore=semaphore,
            file_lock=file_lock,
            start_time=start_time,
            max_runtime=MAX_RUNTIME_SECONDS,
            results_list=results_list,
            output_file=output_file,
            client=client
        )
        active_tasks.append(asyncio.create_task(task_coroutine))
        # Stagger the start of each task by 1 second to avoid simultaneous api bursts
        await asyncio.sleep(1.0)

    # Wait for all launched tasks to complete
    if active_tasks:
        await asyncio.gather(*active_tasks)

    logging.info("Batch evaluation job finished.")
    ModelClient.print_usage_summary()

if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())
    sys.exit(0)