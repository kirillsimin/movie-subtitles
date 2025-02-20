# Movie Subtitle Emotion Analysis

This is a project for analyzing emotional intensity in movie subtitles. Using a transformer-based text classification model, the script extracts emotions from subtitles and generates visualizations to represent emotional trends over time.

The analysis is powered by the j-hartmann/emotion-english-distilroberta-base model from Hugging Face, which classifies text into various emotional categories using a fine-tuned DistilRoBERTa transformer model.

## Files:

- **`chart.py`**: This script processes subtitles and generates emotion graphs for a single movie.
- **`heatmap.py`**: This script generates a heatmap comparing emotional intensity trends across multiple movies.

## How to Use:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run `chart.py` with an `.srt` subtitle file as input:
   ```bash
   python chart.py ./data/subtitles.srt
   ```
3. Run `heatmap.py` to generate a heatmap of emotional intensity across movies in `./data` folder:
   ```bash
   python heatmap.py
   ```

## Results Examples:

### 25 Movies top movies combined
![25-best-movies](https://github.com/user-attachments/assets/0440372a-bf48-4bfd-a0e4-157cbe1e1e2a)

### Interstellar
![interstellar](https://github.com/user-attachments/assets/b0a36400-359b-4286-9a64-3750a54446fc)
