import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import pipeline

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
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()

    # Match each subtitle block
    subtitle_pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)"
    subtitles = re.findall(subtitle_pattern, content, flags=re.S)

    # Extract subtitle details
    parsed_subtitles = []
    for subtitle in subtitles:
        number, start_time, end_time, text = subtitle
        text = text.replace("\n", " ")  # Combine multiline subtitles
        start_seconds = parse_timestamp(start_time.replace(",", "."))
        end_seconds = parse_timestamp(end_time.replace(",", "."))
        parsed_subtitles.append((int(number), start_seconds, end_seconds, text))

    return parsed_subtitles

# Analyze the "neutral" emotion for each subtitle
def analyze_neutral_emotion(subtitles):
    neutral_scores = []
    for _, start_time, end_time, text in subtitles:
        result = classifier(text, top_k=None)
        neutral_score = next((e['score'] for e in result if e['label'] == "neutral"), 0.0)
        emotionality = 1 - neutral_score  # Transform to emotionality
        neutral_scores.append((start_time, end_time, emotionality))
    return neutral_scores

def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Generate heatmap data
def generate_heatmap_data(folder_path, num_time_bins=100):
    movie_names = []
    heatmap_data = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".srt"):
            file_path = os.path.join(folder_path, filename)
            movie_names.append(filename)

            # Parse subtitles
            print(file_path)
            subtitles = parse_srt(file_path)
            total_duration = max(end_time for _, end_time, _ in analyze_neutral_emotion(subtitles))

            # Calculate emotionality scores at each time bin
            neutral_scores = analyze_neutral_emotion(subtitles)
            time_bins = np.linspace(0, total_duration, num_time_bins + 1)
            binned_scores = np.zeros(num_time_bins)

            for start_time, end_time, score in neutral_scores:
                # Find the bin index for this subtitle's time
                bin_index = np.digitize((start_time + end_time) / 2, time_bins) - 1
                if 0 <= bin_index < num_time_bins:
                    binned_scores[bin_index] += score

            # Normalize by the number of overlaps per bin
            binned_counts = np.histogram([((start_time + end_time) / 2) for start_time, end_time, _ in neutral_scores], bins=time_bins)[0]
            binned_scores = np.divide(binned_scores, binned_counts, out=np.zeros_like(binned_scores), where=binned_counts > 0)

            # Set bins with no subtitles to 0.5, but keep actual zeros as is
            binned_scores = np.where((binned_counts == 0) & (binned_scores == 0), 0.2, binned_scores)

            heatmap_data.append(binned_scores)

    return movie_names, np.array(heatmap_data)


def plot_heatmap(movie_names, heatmap_data):
    # Compute the average emotional intensity across movies
    avg_emotionality = np.mean(heatmap_data, axis=0)
    avg_emotionality = smooth(avg_emotionality, window_size=5)

    # Custom colormap with sharper transitions
    cmap = LinearSegmentedColormap.from_list("custom_purple", ["white", "#8A2BE2"])

    plt.figure(figsize=(12, len(movie_names) * 0.5))
    plt.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest')

    # Add horizontal separators for each movie
    for i in range(1, len(movie_names)):
        plt.axhline(i - 0.5, color="black", linewidth=0.5)

    # Overlay average emotional intensity
    plt.plot(
        np.linspace(0, heatmap_data.shape[1] - 1, len(avg_emotionality)),
        (1 - avg_emotionality) * len(movie_names) - 0.5,
        color="red",
        linewidth=2,
        label="Avg Emotional Intensity",
    )

    # Color bar
    cbar = plt.colorbar(label="Emotional Intensity")
    cbar.ax.tick_params(labelsize=10)

    # Adjust ticks and labels
    plt.yticks(range(len(movie_names)), movie_names, fontsize=8)
    plt.xticks(
        np.linspace(0, heatmap_data.shape[1] - 1, 11),
        labels=[f"{int(x)}%" for x in np.linspace(0, 100, 11)],
        fontsize=8,
    )

    # Title and labels
    plt.title("Emotional Intensity Heatmap Across Movies", fontsize=12)
    plt.xlabel("Movie Progression (%)", fontsize=10)
    plt.ylabel("Movies", fontsize=10)
    plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    # Path to the folder containing SRT files
    folder_path = "./data"  # Replace with your folder path

    # Generate heatmap data
    movie_names, heatmap_data = generate_heatmap_data(folder_path)

    # Plot the heatmap
    plot_heatmap(movie_names, heatmap_data)
