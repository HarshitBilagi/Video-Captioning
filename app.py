import os
import json
import sys
import subprocess
import time
import requests
import gdown
import streamlit as st

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="Video Captioning Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to search for video files with multiple extension types
def find_video_path(clip_id):
    extensions = [".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"]
    for ext in extensions:
        candidate = os.path.join("data", "clips", f"{clip_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

# Helper function to render clip caption cards and details
def render_clip_results(clip, idx=1):
    clip_id = clip.get("clip_id", f"Unknown_Clip_{idx}")
    
    st.markdown(f"### 🎬 Clip: {clip_id}")
    
    # Check if video exists and show it
    video_path = find_video_path(clip_id)
    if video_path:
        # Use columns to constrain video width and center it
        v_col1, v_col2, v_col3 = st.columns([1, 0.5, 1])
        with v_col2:
            st.video(video_path)
    else:
        st.info(f"Video file for {clip_id} not found in data/clips/")
        
    # Check for error in clip execution
    if "error" in clip:
        st.error(f"❌ Error during clip processing: {clip['error']}")
        st.markdown("---")
        return
        
    # Display 4 caption boxes side-by-side
    captions = clip.get("captions", {})
    scores = clip.get("self_judge_scores", {})
    
    cols = st.columns(4)
    for col, (tone, caption_text) in zip(cols, captions.items()):
        score_vals = scores.get(tone, {})
        accuracy = score_vals.get("accuracy", 0)
        tone_fit = score_vals.get("tone_fit", 0)
        
        with col:
            st.markdown(f"#### {tone.replace('_', ' ').title()}")
            
            # Check alignment and render matching styling box
            box_content = f"\"{caption_text}\""
            if tone_fit >= 4:
                st.success(box_content)
            elif tone_fit == 3:
                st.warning(box_content)
            else:
                st.error(box_content)
            
            # Accuracy & Tone Fit Small Metrics
            st.write(f"🎯 **Accuracy:** {accuracy}/5 | 🎭 **Tone Fit:** {tone_fit}/5")
            
    # Collapsible Scene Analysis Details
    scene_desc = clip.get("scene_description", {})
    with st.expander("🔍 Show Scene Analysis Details"):
        st.json(scene_desc)
        
    st.markdown("---")

# Page Headers
st.markdown('<div class="main-title">Video Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generating 4-tone captions using Minimax M3 via Fireworks AI</div>', unsafe_allow_html=True)

# Load results.json
results = []
results_exist = os.path.exists("results.json")
if results_exist:
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error reading results.json: {e}")
        results_exist = False

# Load usage.json
usage_data = {}
if os.path.exists("usage.json"):
    try:
        with open("usage.json", "r") as f:
            usage_data = json.load(f)
    except Exception:
        pass

# SIDEBAR: Pipeline Control & Statistics
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    # Re-run Pipeline Button (Full Batch Run)
    if st.button("🚀 Re-run Full Pipeline", use_container_width=True):
        with st.spinner("Running batch pipeline.py..."):
            try:
                # Run pipeline using current virtual env python executable
                process_result = subprocess.run(
                    [sys.executable, "pipeline.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success("Full Pipeline Run Completed!")
                st.toast("Updated results and usage statistics successfully.", icon="✅")
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline process failed with exit code {e.returncode}")
                with st.expander("Show execution log error"):
                    st.code(e.stderr or e.stdout)

    st.markdown("---")
    st.header("📊 Pipeline Statistics")
    
    if results_exist:
        total_clips = len(results)
        st.metric("Total Clips Processed", total_clips)
        
        # Extract total tokens & calls
        total_calls = usage_data.get("total_calls", 0)
        total_tokens = usage_data.get("total_tokens", 0)
        prompt_tokens = usage_data.get("total_prompt_tokens", 0)
        completion_tokens = usage_data.get("total_completion_tokens", 0)
        
        st.metric("Total API Calls", total_calls)
        st.metric("Total Tokens Used", f"{total_tokens:,}")
        
        # Calculate Average Self-Judge Scores
        tone_metrics = {
            t: {"accuracy_sum": 0, "tone_fit_sum": 0, "count": 0}
            for t in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
        }
        
        for clip in results:
            if "error" in clip:
                continue
            scores = clip.get("self_judge_scores", {})
            for tone, s_dict in scores.items():
                if tone in tone_metrics and isinstance(s_dict, dict):
                    tone_metrics[tone]["accuracy_sum"] += s_dict.get("accuracy", 0)
                    tone_metrics[tone]["tone_fit_sum"] += s_dict.get("tone_fit", 0)
                    tone_metrics[tone]["count"] += 1
                    
        st.markdown("### Average Self-Judge Scores")
        for tone, metric in tone_metrics.items():
            count = metric["count"]
            if count > 0:
                avg_acc = metric["accuracy_sum"] / count
                avg_fit = metric["tone_fit_sum"] / count
                st.markdown(f"**{tone.replace('_', ' ').upper()}**")
                st.markdown(f"🎯 Accuracy: `{avg_acc:.2f}/5` | 🎭 Tone Fit: `{avg_fit:.2f}/5`")
            else:
                st.markdown(f"**{tone.replace('_', ' ').upper()}**: `No data`")
    else:
        st.info("Run the pipeline to load stats")

# TAB LAYOUT
tab_dashboard, tab_upload, tab_url = st.tabs(["📊 Results Dashboard", "📤 Upload Video", "🔗 URL / Album Link"])

# TAB 1: Results Dashboard
with tab_dashboard:
    if not results_exist or not results:
        st.warning("⚠️ results.json not found or empty. Click 'Re-run Full Pipeline' in the sidebar or run pipeline.py in your terminal first to generate results.")
    else:
        # Loop over and render each clip
        for idx, clip in enumerate(results, 1):
            render_clip_results(clip, idx)

# TAB 2: Upload Video
with tab_upload:
    st.header("Upload Video Clip")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        dest_path = os.path.join("data", "clips", filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Save uploaded file contents
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Successfully uploaded: `{filename}`")
        
        # Centered video preview
        v_col1, v_col2, v_col3 = st.columns([1, 0.5, 1])
        with v_col2:
            st.video(dest_path)
            
        if st.button("🚀 Run Pipeline on this clip", key="run_pipeline_upload", use_container_width=True):
            with st.spinner(f"Running pipeline on {filename}..."):
                try:
                    subprocess.run(
                        [sys.executable, "pipeline.py", "--clip", filename],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    st.session_state["last_processed_clip"] = filename
                    st.success(f"Pipeline run completed for {filename}!")
                    st.toast(f"Result for {filename} updated.", icon="✅")
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to process clip: {e}")
                    st.code(e.stderr or e.stdout)
                    
        # Render single clip result if processed
        last_processed = st.session_state.get("last_processed_clip")
        if last_processed == filename:
            try:
                with open("results.json", "r") as f:
                    updated_results = json.load(f)
                clip_id = os.path.splitext(filename)[0]
                clip_data = next((c for c in updated_results if c.get("clip_id") == clip_id), None)
                if clip_data:
                    st.markdown("### Processed Clip Results")
                    render_clip_results(clip_data)
                else:
                    st.warning(f"Could not find results for `{clip_id}` in results.json")
            except Exception as e:
                st.error(f"Error loading updated results: {e}")

# TAB 3: URL / Album Link
with tab_url:
    st.header("Download & Process from URL / Album Link")
    url_input = st.text_input("Enter direct video URL or public Google Drive file/folder link", placeholder="https://...")
    
    if url_input:
        is_gdrive = "drive.google.com" in url_input
        is_direct_video = any(ext in url_input.lower() for ext in [".mp4", ".mov", ".avi"])
        
        if not is_gdrive and not is_direct_video:
            st.error("Unsupported URL type. Must be a direct link to a video file (.mp4, .mov, .avi) or a drive.google.com link.")
        else:
            download_clicked = st.button("📥 Download Video", use_container_width=True)
            
            if download_clicked:
                timestamp = int(time.time())
                os.makedirs(os.path.join("data", "clips"), exist_ok=True)
                
                if is_direct_video:
                    # Detect filename from URL or default to mp4 extension
                    url_path = url_input.split("?")[0]
                    file_ext = os.path.splitext(url_path)[1] or ".mp4"
                    filename = f"downloaded_{timestamp}{file_ext}"
                    dest_path = os.path.join("data", "clips", filename)
                    
                    with st.spinner("Downloading direct video..."):
                        try:
                            response = requests.get(url_input, stream=True)
                            response.raise_for_status()
                            total_size = int(response.headers.get('content-length', 0))
                            block_size = 1024 * 1024  # 1 MB
                            
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            
                            written = 0
                            with open(dest_path, "wb") as f:
                                for chunk in response.iter_content(block_size):
                                    f.write(chunk)
                                    written += len(chunk)
                                    if total_size > 0:
                                        percent = min(written / total_size, 1.0)
                                        progress_bar.progress(percent)
                                        status_text.text(f"Downloaded {written / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent*100:.1f}%)")
                                    else:
                                        status_text.text(f"Downloaded {written / (1024*1024):.1f} MB")
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"Downloaded video saved to `{filename}`")
                            st.session_state["downloaded_file_name"] = filename
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                            
                elif is_gdrive:
                    is_folder = "/folders/" in url_input or "/drive/folders/" in url_input
                    
                    if is_folder:
                        with st.spinner("Downloading Google Drive folder..."):
                            try:
                                output_dir = os.path.join("data", "clips")
                                gdown.download_folder(url_input, output=output_dir, quiet=False, remaining_ok=True)
                                st.success("Google Drive folder downloaded to `data/clips/`")
                                st.session_state["downloaded_folder"] = True
                            except Exception as e:
                                st.error(f"Google Drive folder download failed: {e}")
                                st.info("Note: Make sure the Google Drive folder link is public and has 'Anyone with the link can view' permission.")
                    else:
                        filename = f"drive_{timestamp}.mp4"
                        dest_path = os.path.join("data", "clips", filename)
                        with st.spinner("Downloading Google Drive file..."):
                            try:
                                gdown.download(url_input, output=dest_path, quiet=False)
                                st.success(f"Google Drive file saved to `{filename}`")
                                st.session_state["downloaded_file_name"] = filename
                            except Exception as e:
                                st.error(f"Google Drive file download failed: {e}")
                                st.info("Note: Make sure the Google Drive link is public and has 'Anyone with the link can view' permission.")

            # Scan clips directory to offer single-clip executions
            if os.path.exists(os.path.join("data", "clips")):
                all_clips = sorted(os.listdir(os.path.join("data", "clips")))
                video_clips = [f for f in all_clips if f.lower().endswith(('.mp4', '.mov', '.avi'))]
                
                if video_clips:
                    st.markdown("### 🎬 Choose Video to Process")
                    
                    # Set default choice index if download just finished
                    default_idx = 0
                    last_downloaded = st.session_state.get("downloaded_file_name")
                    if last_downloaded in video_clips:
                        default_idx = video_clips.index(last_downloaded)
                        
                    selected_clip = st.selectbox("Select video file", video_clips, index=default_idx)
                    
                    dest_path = os.path.join("data", "clips", selected_clip)
                    
                    # Centered video preview
                    v_col1, v_col2, v_col3 = st.columns([1, 1, 1])
                    with v_col2:
                        st.video(dest_path)
                        
                    if st.button("🚀 Run Pipeline on this clip", key="run_pipeline_url", use_container_width=True):
                        with st.spinner(f"Running pipeline on {selected_clip}..."):
                            try:
                                subprocess.run(
                                    [sys.executable, "pipeline.py", "--clip", selected_clip],
                                    check=True,
                                    capture_output=True,
                                    text=True
                                )
                                st.session_state["last_processed_clip"] = selected_clip
                                st.success(f"Pipeline run completed for {selected_clip}!")
                                st.toast(f"Result for {selected_clip} updated.", icon="✅")
                            except subprocess.CalledProcessError as e:
                                st.error(f"Failed to process clip: {e}")
                                st.code(e.stderr or e.stdout)
                                
                    # Render processed single clip result
                    last_processed = st.session_state.get("last_processed_clip")
                    if last_processed == selected_clip:
                        try:
                            with open("results.json", "r") as f:
                                updated_results = json.load(f)
                            clip_id = os.path.splitext(selected_clip)[0]
                            clip_data = next((c for c in updated_results if c.get("clip_id") == clip_id), None)
                            if clip_data:
                                st.markdown("### Processed Clip Results")
                                render_clip_results(clip_data)
                            else:
                                st.warning(f"Could not find results for `{clip_id}` in results.json")
                        except Exception as e:
                            st.error(f"Error loading updated results: {e}")

# FOOTER SECTION
st.markdown("<br><br>", unsafe_allow_html=True)
footer_cols = st.columns([2, 1, 1])

with footer_cols[0]:
    st.markdown("**Pipeline Architecture:**")
    st.markdown("OpenCV (Frames) ➔ Whisper (Audio) ➔ Minimax M3 (Scene details) ➔ Caption Gen ➔ Self-Judge evaluation")

with footer_cols[1]:
    st.markdown("**API Costs Estimate:**")
    # Rates: $0.30/M input, $1.20/M output
    total_prompt_tokens = usage_data.get("total_prompt_tokens", 0)
    total_completion_tokens = usage_data.get("total_completion_tokens", 0)
    cost = (total_prompt_tokens * 0.30 / 1_000_000) + (total_completion_tokens * 1.20 / 1_000_000)
    st.markdown(f"**Total Cost:** `${cost:.4f}`")
    st.markdown(f"<small>({total_prompt_tokens:,} input | {total_completion_tokens:,} output)</small>", unsafe_allow_html=True)

with footer_cols[2]:
    st.markdown("**Project Link:**")
    st.markdown("[GitHub Repository](https://github.com/HarshitBilagi/Video-Captioning.git)")
