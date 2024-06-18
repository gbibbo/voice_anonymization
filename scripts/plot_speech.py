# invocarlo con la siguiente línea:

# gb0048@datamove1:/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/scripts$ python plot_speech.py /mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Cnn14_DecisionLevelAtt/04/20231119_173000.json /mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/04/20231119_173000.WAV /mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/results/plotspeech_probability_plot.png 219 228 0.2

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import librosa
import librosa.display
import soundfile as sf  # Añadido para guardar el archivo WAV

from matplotlib.ticker import FuncFormatter

# Función para convertir segundos a formato de minutos:segundos
def format_time(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes}:{seconds:02d}"

def process_audio_file(audio_filepath, start_time, end_time, sr=None, output_audio_path=None):
    """Load an audio file, trim it to the specified time interval, and compute its spectrogram."""
    y, sr = librosa.load(audio_filepath, sr=sr, offset=start_time, duration=end_time-start_time)
    S = np.abs(librosa.stft(y))
    S_dB = librosa.power_to_db(S, ref=np.max)
    if output_audio_path:
        #sf.write(output_audio_path, y, sr)  # Guarda el segmento de audio
        print(f"Audio segment saved to {output_audio_path}")
    return S, sr, y, S_dB

def process_and_plot(json_filepath, audio_filepath, output_path, start_time, end_time, threshold):
    """Process the given JSON file and audio file, then plot the spectrogram and speech probability."""
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    filtered_data = [frame for frame in data[:-1] if start_time <= frame['time'] <= end_time]
    times = [frame['time'] for frame in filtered_data]
    speech_probabilities = [next((pred['prob'] for pred in frame['predictions'] if pred['class'] == 'Speech'), 0) for frame in filtered_data]

    output_audio_path = os.path.splitext(output_path)[0] + '_segment.wav'
    S, sr, y, S_dB = process_audio_file(audio_filepath, start_time, end_time, output_audio_path=output_audio_path)
    times_spec = np.linspace(start_time, end_time, S_dB.shape[1])  # Asegura la alineación temporal

    fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    librosa.display.specshow(S_dB, sr=sr, x_axis='time', ax=ax[0], x_coords=times_spec)
    ax[0].set_title('Spectrogram')
    ax[0].set_ylabel('Frequency [Hz]')

    ax[1].plot(times, speech_probabilities, label='Speech Probability', color='b')
    ax[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax[1].set_title('Speech Probability Over Time')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Probability')
    ax[1].legend()
    # Aplicar el formateador al eje X de ambos subgráficos
    ax[0].xaxis.set_major_formatter(FuncFormatter(format_time))
    ax[1].xaxis.set_major_formatter(FuncFormatter(format_time))

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Speech Probability and Spectrogram.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser.add_argument('output_file', type=str, help='Path to save the output plot image.')
    parser.add_argument('start_time', type=float, help='Start time in seconds for the interval to plot.')
    parser.add_argument('end_time', type=float, help='End time in seconds for the interval to plot.')
    parser.add_argument('threshold', type=float, help='Probability threshold for speech detection.')

    args = parser.parse_args()

    process_and_plot(args.json_file, args.audio_file, args.output_file, args.start_time, args.end_time, args.threshold)


