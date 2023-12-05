import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import io
import tempfile

# Function to extract samples from the audio file
def extract_samples(audio_content):
    audio = AudioSegment.from_file(io.BytesIO(audio_content), format="wav", codec="ffmpeg")
    return np.array(audio.get_array_of_samples())

# Function to calculate audio features
def calculate_audio_features(samples):
    features = {
        'Max Amplitude': np.max(np.abs(samples)),
        'Amplitude Std': np.std(samples),
    }
    return features

def evaluate_audio_quality_for_video(video_url, output_folder, sample_rate, frame_size=4410):
    try:
        # Download the video
        video_content = requests.get(video_url).content

        # Extract audio samples
        audio_samples = extract_samples(video_content)

        # List to store results for each frame
        results_for_frames = []

        # Output folder
        video_output_folder = os.path.join(output_folder, "audio_plots")

        # Evaluate audio quality for each frame
        for i in range(0, len(audio_samples), frame_size):
            frame_samples = audio_samples[i:i + frame_size]

            result, glitch_status, glitch_stats, audio_features, plot_filename = evaluate_audio_quality_for_frame(
                frame_samples, i, frame_size, video_output_folder, sample_rate
            )

            results_for_frames.append({
                'Start Time (seconds)': i / sample_rate,  # Adjust the sample rate as needed
                'End Time (seconds)': (i + frame_size) / sample_rate,
                'Glitch Status': glitch_status,
                'Glitch Stats': glitch_stats,
                'Audio Features': audio_features,
                'Plot': plot_filename,
            })

        # Create a DataFrame for the report
        report_df = pd.DataFrame(results_for_frames)

        # Save the report to a temporary file
        excel_file = os.path.join(output_folder, "audio_quality_report_for_frames.xlsx")
        report_df.to_excel(excel_file, index=False)
        print(f"Report saved to {excel_file}")

        return "Audio quality analysis completed!", report_df, excel_file

    except Exception as e:
        return f"Error: {str(e)}", None, None

# Streamlit app code
st.title("No Audio Analysis Demo")

# Git LFS URLs for the videos
original_audio_url = "https://github.com/jyothishridhar/Audio_quality_noaudio/raw/master/referance_audio.wav"
distorted_audio_url = "https://github.com/jyothishridhar/Audio_quality_noaudio/raw/master/testing_audio.wav"

# Download videos
original_video_content = requests.get(original_video_url).content
distorted_video_content = requests.get(distorted_video_url).content

# Add download links
st.markdown(f"**Download Original Video**")
st.markdown(f"[Click here to download the Original Video]({original_video_url})")

st.markdown(f"**Download Distorted Video**")
st.markdown(f"[Click here to download the Distorted Video]({distorted_video_url})")

# Sample rate
sample_rate = 44100

# Add button to run audio quality analysis for the original video
if st.button("Run Audio Quality Analysis (Original)"):
    st.text("Running audio quality analysis for the original video...")
    result_original, report_df_original, excel_file_original = evaluate_audio_quality_for_video(
        original_video_url, tempfile.gettempdir(), sample_rate
    )
    st.success(result_original)

    # Display the DataFrame
    st.dataframe(report_df_original)

    # Display glitch plots
    st.markdown("### Glitch Plots (Original)")
    for i, plot_filename_original in enumerate(report_df_original['Plot']):
        st.image(plot_filename_original, f"Glitch Plot (Original) {i}")

    # Add download link for the report
    st.markdown(f"**Download Audio Quality Report (Original)**")
    st.markdown(f"[Click here to download the Audio Quality Report Excel]({excel_file_original})")

# Add button to run audio quality analysis for the distorted video
if st.button("Run Audio Quality Analysis (Distorted)"):
    st.text("Running audio quality analysis for the distorted video...")
    result_distorted, report_df_distorted, excel_file_distorted = evaluate_audio_quality_for_video(
        distorted_video_url, tempfile.gettempdir(), sample_rate
    )
    st.success(result_distorted)

    # Display the DataFrame
    st.dataframe(report_df_distorted)

    # Display glitch plots
    st.markdown("### Glitch Plots (Distorted)")
    for i, plot_filename_distorted in enumerate(report_df_distorted['Plot']):
        st.image(plot_filename_distorted, f"Glitch Plot (Distorted) {i}")

    # Add download link for the report
    st.markdown(f"**Download Audio Quality Report (Distorted)**")
    st.markdown(f"[Click here to download the Audio Quality Report Excel]({excel_file_distorted})")
