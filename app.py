import os
import json
import sys
import subprocess
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
    
    # Re-run Pipeline Button
    if st.button("🚀 Re-run Pipeline", use_container_width=True):
        with st.spinner("Running batch pipeline.py..."):
            try:
                # Run pipeline using current virtual env python executable
                process_result = subprocess.run(
                    [sys.executable, "pipeline.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success("Pipeline Run Completed!")
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

# MAIN CONTENT AREA
if not results_exist or not results:
    st.warning("⚠️ results.json not found or empty. Click 'Re-run Pipeline' in the sidebar or run pipeline.py in your terminal first to generate results.")
else:
    # Loop over and render each clip
    for idx, clip in enumerate(results, 1):
        clip_id = clip.get("clip_id", f"Unknown_Clip_{idx}")
        
        st.markdown(f"## 🎬 Clip: {clip_id}")
        
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
            continue
            
        # Display 4 caption boxes side-by-side
        captions = clip.get("captions", {})
        scores = clip.get("self_judge_scores", {})
        
        cols = st.columns(4)
        for col, (tone, caption_text) in zip(cols, captions.items()):
            score_vals = scores.get(tone, {})
            accuracy = score_vals.get("accuracy", 0)
            tone_fit = score_vals.get("tone_fit", 0)
            
            with col:
                st.markdown(f"### {tone.replace('_', ' ').title()}")
                
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
