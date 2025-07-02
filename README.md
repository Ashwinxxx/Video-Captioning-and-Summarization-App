# 🎥 Video Captioning and Summarization App

## Project Overview

This project presents a Streamlit-based web application designed to automatically generate descriptive captions for video content and provide a concise summary of the video based on these captions. It leverages state-of-the-art deep learning models for image captioning (specifically, the BLIP model from Salesforce) and integrates with OpenCV for video frame extraction.

The application offers a user-friendly interface for either uploading video files directly or providing a URL to a video file.

## Features

* **Video Input Options:** Upload local video files (MP4, MOV, AVI, MKV) or provide a direct link to a video hosted online.
* **Key Frame Extraction:** Efficiently extracts frames from the video at a configurable interval (frames per second).
* **AI-Powered Image Captioning:** Utilizes the pre-trained `Salesforce/blip-image-captioning-base` model to generate descriptive text captions for each extracted key frame.
* **Video Summarization:** Compiles a basic summary of the video content by combining the generated captions.
* **Visual Output:** Displays the extracted key frames along with their respective captions in a clean grid format.
* **Performance Optimization:** Employs Streamlit's caching mechanisms (`st.cache_resource`) to speed up model loading.
* **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive user experience.
* **Error Handling:** Includes robust error handling for video processing, model loading, and URL fetching.

## How it Works

1.  **Video Input:** The user provides a video either by uploading it or by pasting a direct URL. If a URL is provided, the video is temporarily downloaded.
2.  **Frame Extraction:** OpenCV (`cv2`) is used to open the video file and extract frames at the specified `frame_rate`.
3.  **Caption Generation:** Each extracted frame is converted into a PIL Image and passed to the BLIP model. The model processes the image and generates a textual description (caption). This process is accelerated by running the model on a GPU if available (CUDA).
4.  **Summarization:** A simple summarization technique is applied, currently involving the concatenation of unique generated captions. (This part can be extended with more advanced NLP summarization techniques).
5.  **Display:** The Streamlit app then presents the overall video summary and visualizes each key frame with its corresponding caption.

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository (or download the code):**
    ```bash
    git clone [https://github.com/Ashwinxxx/streamlit-quant-backtester.git](https://github.com/Ashwinxxx/streamlit-quant-backtester.git) # Assuming this is where it will be
    cd streamlit-quant-backtester # Or the directory where you put the files
    ```
    *Note: You might want to create a new, more specific repository like `streamlit-video-captioning`.*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required libraries:**
    Create a file named `requirements.txt` in the root of your project directory with the following content:
    ```
    streamlit
    opencv-python
    torch
    transformers
    pillow
    numpy
    requests
    matplotlib
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the main application file:**
    Ensure your main Python script (e.g., `app.py`) containing the Streamlit code is in the project directory.

## Usage

Once you have installed all the dependencies, you can run the Streamlit application:

```bash
streamlit run app.py
