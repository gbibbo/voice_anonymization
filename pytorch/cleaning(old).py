import json
import argparse
import os
import numpy as np
import librosa
import soundfile as sf

def read_json_and_get_indices(json_path, label='Speech', threshold=0.4):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extracción de los metadatos y ajuste de la estructura de datos
    metadata = data[-1]['metadata']
    hop_size = metadata['hop_size']
    window_size = metadata['window_size']
    detection_results = data[:-1]

    # Encuentra los índices de frames donde la etiqueta "Speech" supera el umbral de confianza
    speech_frames = []
    frame_mask = [1] * len(detection_results)  # Initialize mask with 1s (no speech detected by default)

    for i, entry in enumerate(detection_results):
        frame_contains_speech = False
        for prediction in entry['predictions']:
            if prediction['class_label'] == label and prediction['probability'] > threshold:
                frame_contains_speech = True
                break
        if frame_contains_speech:
            speech_frames.append(entry['frame_index'] * hop_size)
            frame_mask[i] = 0  # Mark this frame as containing speech

    return speech_frames, frame_mask, hop_size, window_size

def generate_low_amplitude_noise(length, amplitude=1e-10):
    return np.random.normal(0, amplitude, length)

def mute_speech_sections(audio_path, speech_frames, hop_size, window_size):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Apply noise to speech segments
    for start in speech_frames:
        end = start + window_size
        if end > len(audio):
            end = len(audio)
        audio[start:end] = generate_low_amplitude_noise(end - start)

    # Save the modified audio
    modified_audio_path = os.path.splitext(audio_path)[0] + '_modified.wav'
    sf.write(modified_audio_path, audio, sr)
    print(f'Audio modificado guardado en: {modified_audio_path}')

    return modified_audio_path

def create_mask_json(original_json_path, mask, output_json_path):
    with open(original_json_path, 'r') as f:
        data = json.load(f)

    detection_results = data[:-1]
    new_detection_results = []

    for i, entry in enumerate(detection_results):
        new_entry = {
            'frame_index': entry['frame_index'],
            'sample_index': entry['sample_index'],
            'time': entry['time'],
            'mask': mask[i]
        }
        new_detection_results.append(new_entry)

    # Save the modified JSON with mask
    with open(output_json_path, 'w') as f:
        json.dump(new_detection_results + [data[-1]], f, indent=4)  # Keep the metadata unchanged
    print(f'Mask JSON guardado en: {output_json_path}')

def main(json_filename):
    json_path = os.path.join('results', json_filename)
    audio_filename = os.path.splitext(json_filename)[0] + '.wav'
    audio_path = os.path.join('resources', audio_filename)
    
    speech_frames, frame_mask, hop_size, window_size = read_json_and_get_indices(json_path)
    modified_audio_path = mute_speech_sections(audio_path, speech_frames, hop_size, window_size)
    
    output_json_path = os.path.splitext(modified_audio_path)[0] + '_mask.json'
    create_mask_json(json_path, frame_mask, output_json_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Elimina secciones de 'Speech' de un archivo de audio y genera máscara JSON.")
    parser.add_argument('json_filename', type=str, help='Nombre del archivo JSON de entrada con resultados de detección.')
    
    args = parser.parse_args()
    
    main(args.json_filename)
