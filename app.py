import streamlit as st
import os
from pipeline import run_pipeline

def main():
    st.set_page_config(page_title="Video Captioning Pipeline", layout="wide")
    st.title("🎥 Video Captioning Pipeline")
    
    st.sidebar.header("Configuration")
    tone = st.sidebar.selectbox(
        "Select Caption Tone",
        options=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    )
    
    uploaded_file = st.file_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save temporary file
        temp_dir = "data/clips"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.video(video_path)
        
        if st.button("Run Captioning Pipeline"):
            with st.spinner("Processing video..."):
                result = run_pipeline(video_path, tone)
                
            st.success("Pipeline execution complete!")
            st.json(result)

if __name__ == "__main__":
    main()
