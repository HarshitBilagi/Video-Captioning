import os
import json
import glob
import time
import logging
from typing import Dict, Any, List

# Setup stage logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    from src.extraction import extract_frames, extract_transcript
    from src.scene_understanding import describe_scene
    from src.caption_generation import generate_all_captions
    from src.self_judge import judge_all_captions
    from src.model_client import ModelClient
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from extraction import extract_frames, extract_transcript
    from scene_understanding import describe_scene
    from caption_generation import generate_all_captions
    from self_judge import judge_all_captions
    from model_client import ModelClient

def run_batch_pipeline(clips_dir: str = "data/clips", output_file: str = "results.json", target_clip: str = None) -> None:
    """
    Orchestrates the video captioning pipeline across all video clips in the clips directory.
    Saves the final output JSON and prints a complete summary at the end.
    """
    # Reset usage counters for this run
    ModelClient.reset_usage()

    # Verify/create directory structures
    os.makedirs(clips_dir, exist_ok=True)

    if target_clip:
        # Check if the target clip path is absolute or exists relative to the CWD,
        # otherwise look inside the clips_dir
        if os.path.isabs(target_clip) or os.path.exists(target_clip):
            video_files = [target_clip]
        else:
            video_files = [os.path.join(clips_dir, target_clip)]
    else:
        # Search for clips with supported extensions
        extensions = ("*.mp4", "*.mov", "*.avi")
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(clips_dir, ext)))
            # Also support lowercase extensions or windows glob differences
            video_files.extend(glob.glob(os.path.join(clips_dir, ext.upper())))

    # De-duplicate and sort paths
    video_files = sorted(list(set(video_files)))
    total_clips = len(video_files)

    print("==========================================")
    print("      VIDEO CAPTIONING PIPELINE RUN       ")
    print("==========================================")
    print(f"Clips directory: {clips_dir}")
    print(f"Output file:     {output_file}")
    print(f"Found {total_clips} video files to process.")
    print("==========================================\n")

    results = []
    # If processing a single target clip, load existing results to merge/update
    if target_clip and os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                results = json.load(f)
                if not isinstance(results, list):
                    results = []
        except Exception:
            results = []

    success_count = 0
    failure_count = 0

    # Registry for storing judgment scores to calculate average
    score_registry = {
        "formal": {"accuracy": [], "tone_fit": []},
        "sarcastic": {"accuracy": [], "tone_fit": []},
        "humorous_tech": {"accuracy": [], "tone_fit": []},
        "humorous_non_tech": {"accuracy": [], "tone_fit": []}
    }

    for idx, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        clip_id = os.path.splitext(filename)[0]

        print(f"[{idx}/{total_clips}] Starting processing for: {filename}")
        
        try:
            # Stage 1: Extraction
            print("  [Stage 1/4] Extracting frames & audio transcripts...")
            frames = extract_frames(video_path, fps_sample=1.0, max_frames=8)
            transcript = extract_transcript(video_path)

            # Stage 2: Scene Understanding
            print("  [Stage 2/4] Generating structured scene understanding...")
            scene_json = describe_scene(frames, transcript)

            # Stage 3: Caption Generation
            print("  [Stage 3/4] Generating captions for all tones...")
            captions = generate_all_captions(scene_json)

            # Stage 4: Self-Judgment
            print("  [Stage 4/4] Evaluating captions against intended tones...")
            judge_scores = judge_all_captions(captions, scene_json)

            print(f"  [SUCCESS] Successfully completed processing {filename}\n")

            # Assemble successful results dict
            clip_result = {
                "clip_id": clip_id,
                "scene_description": scene_json,
                "captions": captions,
                "self_judge_scores": judge_scores
            }
            if target_clip:
                # Remove existing entries with this clip_id to prevent duplicates
                results = [r for r in results if r.get("clip_id") != clip_id]
            results.append(clip_result)
            success_count += 1

            # Save scores for statistics
            for tone, s in judge_scores.items():
                if tone in score_registry:
                    score_registry[tone]["accuracy"].append(s.get("accuracy", 0))
                    score_registry[tone]["tone_fit"].append(s.get("tone_fit", 0))

        except Exception as e:
            # Log warning, capture failure, and resume next clip
            logging.error(f"Error processing clip {filename}: {e}", exc_info=True)
            print(f"  [ERROR] Failed processing {filename}. Error: {e}\n")

            clip_result = {
                "clip_id": clip_id,
                "error": str(e)
            }
            if target_clip:
                results = [r for r in results if r.get("clip_id") != clip_id]
            results.append(clip_result)
            failure_count += 1

        # Pause between clips to avoid rate limits on the free tier
        if idx < total_clips:
            print("  [WAIT] Pausing 3 seconds before next clip...")
            time.sleep(3)

    # Write per-clip results output
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Pipeline results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving pipeline results JSON: {e}")

    # Summary report
    print("\n==========================================")
    print("            PIPELINE RUN SUMMARY")
    print("==========================================")
    print(f"Total Clips Processed: {total_clips}")
    print(f"Successes:            {success_count}")
    print(f"Failures:             {failure_count}")

    if success_count > 0:
        print("\nAverage Self-Judge Scores (1-5 scale):")
        for tone, metrics in score_registry.items():
            accs = metrics["accuracy"]
            fits = metrics["tone_fit"]
            
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            avg_fit = sum(fits) / len(fits) if fits else 0.0
            
            print(f"  {tone.upper()}:")
            print(f"    - Accuracy: {avg_acc:.2f}/5.00")
            print(f"    - Tone Fit: {avg_fit:.2f}/5.00")
    print("==========================================")

    # Print API usage summary
    ModelClient.print_usage_summary()

    # Save usage stats to a file for Streamlit UI ingestion
    usage_data = {
        "total_calls": ModelClient._total_calls,
        "total_prompt_tokens": ModelClient._total_prompt_tokens,
        "total_completion_tokens": ModelClient._total_completion_tokens,
        "total_tokens": ModelClient._total_tokens,
        "max_calls_limit": ModelClient._MAX_CALLS_PER_RUN
    }
    try:
        with open("usage.json", "w") as uf:
            json.dump(usage_data, uf, indent=2)
        print("Usage summary saved to: usage.json")
    except Exception as e:
        print(f"Error saving usage summary to usage.json: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Video Captioning Pipeline")
    parser.add_argument("--clip", type=str, default=None, help="Process only a specific clip filename")
    parser.add_argument("--clips-dir", type=str, default="data/clips", help="Directory containing clips")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save results JSON")
    args = parser.parse_args()

    run_batch_pipeline(clips_dir=args.clips_dir, output_file=args.output, target_clip=args.clip)
