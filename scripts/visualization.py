import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def load_predictions(json_path):
    """Loads the sound event predictions from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data[:-1]  # Excludes metadata at the end

def extract_top_categories(data, top_n=5):
    """Extracts the most representative categories based on the sum of probabilities."""
    category_prob = {}
    for entry in data:
        for prediction in entry['predictions']:
            if prediction['class'] in category_prob:
                category_prob[prediction['class']] += prediction['prob']
            else:
                category_prob[prediction['class']] = prediction['prob']
    # Sort and select the top_n categories
    top_categories = sorted(category_prob, key=category_prob.get, reverse=True)[:top_n]
    return top_categories

def create_plots(audio_path, json_path, muted_path, sr=48000, output_dir='output_plots'):
    """Generates and saves a combined plot of the audio spectrogram, sound event predictions, and muted speech spectrogram."""
    # Load data
    audio, _ = librosa.load(audio_path, sr=sr)
    muted_audio, _ = librosa.load(muted_path, sr=sr)
    predictions = load_predictions(json_path)
    top_categories = extract_top_categories(predictions)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a combined plot
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)  # Share x-axis, 3 rows now
    
    # Plot original spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title('Audio Spectrogram')

    # Plot sound event predictions
    times = [entry['time'] for entry in predictions]
    for category in top_categories:
        probs = [next((p['prob'] for p in entry['predictions'] if p['class'] == category), 0) for entry in predictions]
        axs[1].plot(times, probs, label=category)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Probability')
    axs[1].set_title('Sound Event Predictions')
    axs[1].axhline(y=0.2, color='red', linestyle='--')  # Add horizontal line at threshold
    axs[1].legend()

    # Plot muted speech spectrogram
    M = librosa.feature.melspectrogram(y=muted_audio, sr=sr)
    M_dB = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_dB, sr=sr, x_axis='time', y_axis='mel', ax=axs[2])
    axs[2].set_title('Spectrogram with Speech Removed')

    # Set consistent limits and layout
    plt.xlim([times[0], times[-1]])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plot.png'))  # Save as a single image

    plt.close(fig)  # Close the figure to free memory

# Usage of the function
audio_path = '/content/voice_anonymization/resources/sync.WAV'
json_path = '/content/voice_anonymization/resources/Cnn14_DecisionLevelAtt/sync.json'
muted_path = '/content/voice_anonymization/resources/sync_muted.WAV'
create_plots(audio_path, json_path, muted_path)

