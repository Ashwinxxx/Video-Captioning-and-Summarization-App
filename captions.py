import streamlit as st
import tempfile
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from PIL import Image
import numpy as np

# Initialize model and processor (cache to avoid reloading on every run)
@st.cache_resource
def load_blip_model():
    st.info("Loading BLIP model and processor... This may take a moment.")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        st.success(f"✅ BLIP model loaded successfully on {device}!")
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Error loading BLIP model or processor: {str(e)}")
        return None, None, None

# Frame extraction function
def extract_key_frames(video_path, fps_rate):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ Error: Could not open video at {video_path}. Please check the file path or URL.")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(video_fps / fps_rate) if fps_rate > 0 else 1

        frame_count = 0
        pbar = st.progress(0, text="ℹ️ Extracting key frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % max(1, interval) == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
            frame_count += 1
            pbar.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        pbar.empty()
        st.success(f"✅ Extracted {len(frames)} key frames.")
    except Exception as e:
        st.error(f"❌ Frame extraction error: {str(e)}")
    return frames

# Caption generation
def generate_captions(frames, processor, model, device):
    captions = []
    pbar = st.progress(0, text="Generating Captions...")

    for idx, frame in enumerate(frames):
        inputs = processor(images=frame, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
        pbar.progress((idx + 1) / len(frames))

    pbar.empty()
    return captions

# Video download helper
def download_video_from_url(url):
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        st.error("❌ Invalid URL format. Please ensure it starts with http:// or https://")
        return None

    try:
        st.info("ℹ️ Attempting to download video from URL...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(temp_file.name, 'wb') as f:
            for data in tqdm(response.iter_content(1024), total=total_size // 1024, unit='KB'):
                f.write(data)

        st.success(f"✅ Video downloaded successfully! Temporary path: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        st.error(f"❌ Error downloading video from URL: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("🎥 Video Captioning and Summarization with BLIP")
    st.markdown("""
    This application extracts key frames from a video (uploaded or via URL), generates captions for each frame using the Salesforce BLIP model, and provides a basic summary of the video content.
    """)

    processor, model, device = load_blip_model()
    if not processor or not model:
        st.stop()

    st.markdown("---")
    video_source = st.radio("Choose video source:", ("Upload Video File", "Enter Video URL"))

    video_path = None
    if video_source == "Upload Video File":
        uploaded_file = st.file_uploader("Upload a video file (e.g., .mp4, .mov)", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                st.success(f"✅ Video uploaded successfully! Temporary path: {video_path}")
    else:
        video_url = st.text_input("Paste video URL (e.g., direct link to .mp4 file)")
        if video_url:
            video_path = download_video_from_url(video_url)

    st.markdown("---")
    fps_rate = st.slider("Select Frame Extraction Rate (frames per second):", 0.5, 5.0, 1.0)

    if st.button("Generate Captions and Summary"):
        if not video_path or not os.path.exists(video_path):
            st.info("ⓘ Please upload a video file or enter a valid video URL to start.")
            st.stop()

        frames = extract_key_frames(video_path, fps_rate)
        if not frames:
            st.warning("⚠️ No frames could be extracted from the video. Please check the video file or URL.")
            st.stop()

        captions = generate_captions(frames, processor, model, device)

        # Display Summary
        summary_text = " ".join(list(set(captions)))
        st.markdown("### Video Summary:")
        st.info(summary_text)

        # Display Key Frames with Captions
        st.markdown("### Key Frames with Captions:")
        cols = st.columns(3)
        for idx, (frame, caption) in enumerate(zip(frames, captions)):
            with cols[idx % 3]:
                st.image(np.array(frame), caption=f"Frame {idx + 1}: {caption}", use_column_width=True)

if __name__ == "__main__":
    main()
