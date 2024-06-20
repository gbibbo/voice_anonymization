import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import argparse

def load_predictions(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data[:-1]  # Excludes metadata at the end

def extract_top_categories(data, top_n=5):
    category_prob = {}
    for entry in data:
        for prediction in entry['predictions']:
            if prediction['class'] in category_prob:
                category_prob[prediction['class']] += prediction['prob']
            else:
                category_prob[prediction['class']] = prediction['prob']
    top_categories = sorted(category_prob, key=category_prob.get, reverse=True)[:top_n]
    return top_categories

def create_plots(audio_path, json_path, muted_path, sr=48000, output_dir='output_plots'):
    audio, _ = librosa.load(audio_path, sr=sr)
    muted_audio, _ = librosa.load(muted_path, sr=sr)
    predictions = load_predictions(json_path)
    top_categories = extract_top_categories(predictions)
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title('Audio Spectrogram')
    times = [entry['time'] for entry in predictions]
    for category in top_categories:
        probs = [next((p['prob'] for p in entry['predictions'] if p['class'] == category), 0) for entry in predictions]
        axs[1].plot(times, probs, label=category)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Probability')
    axs[1].set_title('Sound Event Predictions')
    axs[1].legend()
    M = librosa.feature.melspectrogram(y=muted_audio, sr=sr)
    M_dB = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_dB, sr=sr, x_axis='time', y_axis='mel', ax=axs[2])
    axs[2].set_title('Spectrogram with Speech Removed')
    plt.xlim([times[0], times[-1]])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plot.png'))
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates visualizations for sound event detection.")
    parser.add_argument('audio_path', type=str, help='Path to the original audio file.')
    parser.add_argument('json_path', type=str, help='Path to the JSON file with detection results.')
    parser.add_argument('muted_path', type=str, help='Path to the muted audio file.')
    parser.add_argument('--sample_rate', type=int, default=48000, help='Sample rate of the audio files.')
    parser.add_argument('--output_dir', type=str, default='output_plots', help='Directory to save the output plots.')
    args = parser.parse_args()
    create_plots(args.audio_path, args.json_path, args.muted_path, args.sample_rate, args.output_dir)
