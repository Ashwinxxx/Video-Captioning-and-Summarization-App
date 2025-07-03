import streamlit as st
import tempfile
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
from urllib.parse import urlparse
import requests
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Video Captioning with BLIP",
    page_icon="🎥",
    layout="wide"
)

# Initialize model and processor (cache to avoid reloading on every run)
@st.cache_resource
def load_blip_model():
    """Load BLIP model and processor with error handling"""
    with st.spinner("Loading BLIP model and processor... This may take a moment."):
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            st.success(f"✅ BLIP model loaded successfully on {device}!")
            return processor, model, device
        except Exception as e:
            st.error(f"❌ Error loading BLIP model: {str(e)}")
            st.info("💡 This might be due to insufficient resources. Try using a smaller model or running locally.")
            return None, None, None

def extract_key_frames(video_path, fps_rate):
    """Extract key frames from video with improved error handling"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ Error: Could not open video file. Please check the file format.")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps <= 0:
            st.error("❌ Error: Could not determine video FPS. The video file might be corrupted.")
            cap.release()
            return frames
        
        interval = max(1, int(video_fps / fps_rate))
        
        st.info(f"📊 Video info: {total_frames} total frames, {video_fps:.2f} FPS")
        
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
            
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Extracting frames... {frame_count}/{total_frames}")

        cap.release()
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Extracted {len(frames)} key frames from video.")
        
    except Exception as e:
        st.error(f"❌ Frame extraction error: {str(e)}")
    
    return frames

def generate_captions(frames, processor, model, device):
    """Generate captions for frames with progress tracking"""
    captions = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for idx, frame in enumerate(frames):
            status_text.text(f"Generating caption for frame {idx + 1}/{len(frames)}...")
            
            # Process image
            inputs = processor(images=frame, return_tensors="pt").to(device)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=4)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
            
            progress = (idx + 1) / len(frames)
            progress_bar.progress(progress)
            
    except Exception as e:
        st.error(f"❌ Caption generation error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

    return captions

def download_video_from_url(url):
    """Download video from URL with better error handling"""
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        st.error("❌ Invalid URL format. Please ensure it starts with http:// or https://")
        return None

    try:
        st.info("🔄 Downloading video from URL...")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        # Download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size > 100 * 1024 * 1024:  # 100MB limit
            st.warning("⚠️ Video file is quite large. This might take a while...")
        
        downloaded = 0
        progress_bar = st.progress(0)
        
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)
        
        progress_bar.empty()
        st.success("✅ Video downloaded successfully!")
        return temp_file.name
        
    except requests.RequestException as e:
        st.error(f"❌ Error downloading video: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in st.session_state.get('temp_files', []):
            if os.path.exists(file):
                os.unlink(file)
    except:
        pass

def main():
    st.title("🎥 Video Captioning with BLIP")
    st.markdown("""
    This application extracts key frames from videos and generates captions using the Salesforce BLIP model.
    
    **Features:**
    - Upload video files or provide URLs
    - Adjustable frame extraction rate
    - AI-powered caption generation
    - Visual summary of video content
    """)

    # Initialize session state
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []

    # Load model
    processor, model, device = load_blip_model()
    if not processor or not model:
        st.error("❌ Failed to load the BLIP model. Please check your internet connection and try again.")
        st.stop()

    st.markdown("---")
    
    # Video source selection
    video_source = st.radio(
        "Choose video source:",
        ("Upload Video File", "Enter Video URL"),
        help="Select how you want to provide the video"
    )

    video_path = None
    
    if video_source == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "wmv"],
            help="Supported formats: MP4, MOV, AVI, MKV, WMV"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                st.session_state.temp_files.append(video_path)
                st.success(f"✅ Video uploaded successfully! ({uploaded_file.size / 1024 / 1024:.1f} MB)")
    else:
        video_url = st.text_input(
            "Enter video URL",
            placeholder="https://example.com/video.mp4",
            help="Direct link to video file (must end with video extension)"
        )
        
        if video_url:
            if st.button("Download Video"):
                video_path = download_video_from_url(video_url)
                if video_path:
                    st.session_state.temp_files.append(video_path)

    # Frame extraction settings
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        fps_rate = st.slider(
            "Frame extraction rate (frames per second):",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Lower values = fewer frames, faster processing"
        )
    
    with col2:
        max_frames = st.number_input(
            "Maximum frames to extract:",
            min_value=5,
            max_value=50,
            value=20,
            help="Limit total frames to control processing time"
        )

    # Processing button
    if st.button("🚀 Generate Captions", type="primary"):
        if not video_path or not os.path.exists(video_path):
            st.warning("⚠️ Please upload a video file or download from URL first.")
            st.stop()

        with st.spinner("Processing video..."):
            # Extract frames
            frames = extract_key_frames(video_path, fps_rate)
            
            if not frames:
                st.error("❌ No frames could be extracted from the video.")
                st.stop()
            
            # Limit frames if needed
            if len(frames) > max_frames:
                frames = frames[:max_frames]
                st.info(f"ℹ️ Limited to {max_frames} frames for processing efficiency.")

            # Generate captions
            captions = generate_captions(frames, processor, model, device)
            
            if not captions:
                st.error("❌ Failed to generate captions.")
                st.stop()

        # Display results
        st.markdown("---")
        st.markdown("## 📋 Results")
        
        # Summary
        with st.expander("📝 Video Summary", expanded=True):
            unique_captions = list(set(captions))
            summary_text = " ".join(unique_captions)
            st.write(summary_text)
            
            # Word frequency
            words = summary_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            if word_freq:
                st.markdown("**Key themes:**")
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                st.write(", ".join([f"{word} ({count})" for word, count in top_words]))

        # Frame gallery
        st.markdown("### 🖼️ Key Frames with Captions")
        
        # Create columns for gallery
        cols = st.columns(3)
        for idx, (frame, caption) in enumerate(zip(frames, captions)):
            with cols[idx % 3]:
                st.image(
                    np.array(frame),
                    caption=f"Frame {idx + 1}: {caption}",
                    use_column_width=True
                )
        
        # Download results
        st.markdown("---")
        if st.button("📥 Download Results as Text"):
            results_text = f"Video Captioning Results\n{'='*50}\n\n"
            results_text += f"Summary:\n{summary_text}\n\n"
            results_text += "Individual Frame Captions:\n"
            for idx, caption in enumerate(captions):
                results_text += f"Frame {idx + 1}: {caption}\n"
            
            st.download_button(
                label="Download Results",
                data=results_text,
                file_name="video_captions.txt",
                mime="text/plain"
            )

    # Cleanup on app shutdown
    if st.button("🧹 Clean Up Temporary Files"):
        cleanup_temp_files()
        st.success("✅ Temporary files cleaned up!")

if __name__ == "__main__":
    main()
