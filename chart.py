import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import pipeline
from pprint import pprint

# Initialize the emotion classifier
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=0  # Set to -1 if you want to use CPU
)

# Function to parse timestamps into seconds
def parse_timestamp(timestamp):
    hours, minutes, seconds = map(float, re.split("[:,]", timestamp))
    return hours * 3600 + minutes * 60 + seconds

# Function to parse the SRT file
def parse_srt(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Match each subtitle block
    subtitle_pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)"
    subtitles = re.findall(subtitle_pattern, content, flags=re.S)

    # Extract subtitle details
    parsed_subtitles = []
    for subtitle in subtitles:
        number, start_time, end_time, text = subtitle
        # Combine multiline subtitles
        text = text.replace("\n", " ")
        start_seconds = parse_timestamp(start_time.replace(",", "."))
        end_seconds = parse_timestamp(end_time.replace(",", "."))
        parsed_subtitles.append((int(number), start_seconds, end_seconds, text))

    return parsed_subtitles

# Analyze emotions in each subtitle
def analyze_emotions(subtitles):
    subtitle_emotions = []
    for number, start_time, end_time, text in subtitles:
        result = classifier(text, top_k=None)  # Get all emotion scores
        subtitle_emotions.append((number, start_time, end_time, text, result))
    return subtitle_emotions

# Moving average function
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Plot emotions over time with smoothing
def plot_smoothed_emotions(subtitle_emotions, total_duration, window_size=10, title="Subtitles Emotion Strength"):
    # Prepare data for plotting
    emotion_scores = {}
    time_percentages = [((start_time + end_time) / 2) / total_duration * 100 for _, start_time, end_time, _, _ in subtitle_emotions]

    for _, _, _, _, emotions in subtitle_emotions:
        for emotion in emotions:
            label = emotion['label']
            score = emotion['score']
            if label not in emotion_scores:
                emotion_scores[label] = []
            emotion_scores[label].append(score)

    # Define consistent colors for each emotion
    emotion_colors = {
        "joy": "gold",
        "sadness": "blue",
        "anger": "red",
        "fear": "purple",
        "surprise": "green",
        "disgust": "brown",
        "neutral": "gray"
    }

    # Sort labels: "neutral" first, then alphabetical order of remaining emotions
    sorted_labels = ["neutral"] + sorted([label for label in emotion_scores if label != "neutral"])

    # Plot each emotion
    plt.figure(figsize=(12, 6))
    for label in sorted_labels:
        if label in emotion_scores:
            smoothed_scores = moving_average(emotion_scores[label], window_size)
            smoothed_times = moving_average(time_percentages, window_size)
            plt.plot(smoothed_times, smoothed_scores, label=label, color=emotion_colors.get(label, "gray"))

    # Chart labels and legend
    plt.title(title)
    plt.xlabel("Movie Progression (%)")
    plt.ylabel("Emotion Strength")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze emotions in subtitle files")
    parser.add_argument("path", type=str, help="Path to the SRT file")
    args = parser.parse_args()

    srt_file_path = args.path
    title = "Emotions over Time"
    subtitles = parse_srt(srt_file_path)

    total_duration = max([end_time for _, _, end_time, _ in subtitles])
    subtitle_emotions = analyze_emotions(subtitles)
    pprint(subtitle_emotions)
    plot_smoothed_emotions(subtitle_emotions, total_duration, window_size=100, title=title)

