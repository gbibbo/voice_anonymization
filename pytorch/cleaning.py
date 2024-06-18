import json
import argparse
import numpy as np
import librosa
import soundfile as sf

def read_json_and_get_indices(json_path, labels, threshold=0.2, sr=48000):
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data[-1]['metadata']
    hop_size = metadata['hop_size']
    window_size = metadata['window_size']
    detection_results = data[:-1]

    speech_intervals = []

    for i, entry in enumerate(detection_results):
        current_frame = entry['frame_index']
        current_time = current_frame * hop_size / sr

        # Analizar si hay detecciones válidas y guardar los intervalos de inicio y fin extendidos
        for prediction in entry['predictions']:
            if prediction['class'] in labels and prediction['prob'] > threshold:
                # Añadir o ajustar el intervalo de silencio actual
                start_time = max(current_time - 1, 0)  # Extender un segundo antes
                end_time = current_time + (window_size / sr) + 1  # Extender un segundo después

                # Fusionar con el intervalo anterior si es necesario
                if speech_intervals and speech_intervals[-1][1] >= start_time:
                    speech_intervals[-1][1] = max(speech_intervals[-1][1], end_time)
                else:
                    speech_intervals.append([start_time, end_time])

    return speech_intervals, hop_size, window_size

def mute_speech_sections(audio_path, output_path, speech_intervals, sr):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Convertir tiempos a muestras
    modifications = [(int(start * sr), int(end * sr)) for start, end in speech_intervals]

    # Aplicar las modificaciones
    for start, end in modifications:
        if end > len(audio):
            end = len(audio)
        audio[start:end] = generate_low_amplitude_noise(end - start)

    sf.write(output_path, audio, sr)
    print(f'\nModified audio saved at: {output_path}')

def generate_low_amplitude_noise(length, amplitude=1e-10):
    return np.random.normal(0, amplitude, length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process audio by removing sections where speech or singing is detected.")
    parser.add_argument('json_path', type=str, help='Path to the JSON file with detection results.')
    parser.add_argument('audio_input', type=str, help='Path to the original audio file.')
    parser.add_argument('audio_output', type=str, help='Path where the modified audio file should be saved.')
    args = parser.parse_args()

    _, sr = librosa.load(args.audio_input, sr=None, mono=True)
    labels = ["Speech", "Singing", "Male singing", "Female singing", "Child singing", "Male speech, man speaking", "Female speech, woman speaking", "Conversation", "Narration, monologue", "Music"]

    speech_intervals, hop_size, window_size = read_json_and_get_indices(args.json_path, labels, sr=sr)
    mute_speech_sections(args.audio_input, args.audio_output, speech_intervals, sr)

