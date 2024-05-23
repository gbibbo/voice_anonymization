import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import json

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config


def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


def sound_event_detectionORIGINAL(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    # -------------------------------------------------------------------------------------
    # ------ CREAR ARCHIVO .JSON ----------------------------------------------------------

    # Obtener las 7 principales estimaciones de probabilidad para cada frame
    top_k = 7
    top_probabilities = np.argsort(framewise_output, axis=1)[:, -top_k:][:, ::-1]  # Indices de las clases

    # Crear una estructura de datos para almacenar en JSON
    detection_results = []
    for frame_index, top_classes in enumerate(top_probabilities):
        frame_results = {
            'frame_index': frame_index,
            'sample_index': frame_index * hop_size,
            'time': frame_index * hop_size / sample_rate,
            'predictions': []
        }
        for class_index in top_classes:
            # Convertir el valor de float32 a float antes de agregarlo a la estructura de datos
            probability = float(framewise_output[frame_index, class_index])
            frame_results['predictions'].append({
                'class_label': labels[class_index],
                'probability': probability
            })
        detection_results.append(frame_results)

    # Agregar metadatos al final del JSON
    metadata = {
        'sample_rate': sample_rate,
        'hop_size': hop_size,
        'window_size': window_size
    }
    detection_results.append({'metadata': metadata})


    # Guardar los resultados en un archivo JSON
    json_path = os.path.join('results', '{}.json'.format(get_filename(audio_path)))
    with open(json_path, 'w') as json_file:
        json.dump(detection_results, json_file, indent=4)

    print('Saved sound event detection results to {}'.format(json_path))

    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels

def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('  GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device) # Convert to tensor 

    segment_duration = 600  # seconds
    total_duration_seconds = waveform.shape[1] / sample_rate
    num_segments = int(np.ceil(total_duration_seconds / segment_duration))
    print(f"  Total audio duration: {total_duration_seconds:.2f} seconds")
    print(f"  Number of segments: {num_segments:.2f}")

    all_detection_results = []

    # Process each segment of audio
    for segment in range(num_segments):
        print(f"  Segment {segment + 1}")
        start_sample = int(segment * segment_duration * sample_rate)
        end_sample = int(min((segment + 1) * segment_duration * sample_rate, waveform.shape[1]))

        if end_sample <= start_sample:
            print(f"    Segment {segment + 1} is empty, skipping.")
            continue

        waveform_segment = waveform[:, start_sample:end_sample]

        if waveform_segment.size == 0:
            print(f"    Empty segment from {start_sample} to {end_sample}, skipping.")
            continue

        print(f"    Processing from {start_sample / sample_rate:.2f} to {end_sample / sample_rate:.2f} seconds")

        # Forward
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform_segment, None)

        if 'framewise_output' not in batch_output_dict or batch_output_dict['framewise_output'].nelement() == 0:
            print(f"  No output for segment {segment + 1}, possibly due to insufficient data.")
            continue        

        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""

        print('    Sound event detection result (time_steps x classes_num): {}'.format(framewise_output.shape))

        # ------ Create .JSON ----------------------------------------------------------

        top_k = 7
        for frame_index, frame_output in enumerate(framewise_output):
            frame_results = {
                'frame_index': frame_index + (start_sample // hop_size),
                'sample_index': start_sample + frame_index * hop_size,
                'time': (start_sample + frame_index * hop_size) / sample_rate,
                'predictions': []
            }
            top_classes = np.argsort(frame_output)[-top_k:][::-1]
            for class_index in top_classes:
                probability = round(float(frame_output[class_index]), 4)
                frame_results['predictions'].append({
                    'class': labels[class_index],
                    'prob': probability
                })
            all_detection_results.append(frame_results)

    # Extracts the recorder number
    path_parts = audio_path.split(os.sep)
    recorder_number = path_parts[-2]

    # Metadata for the whole file
    metadata = {
        'sample_rate': args.sample_rate,
        'hop_size': args.hop_size,
        'window_size': args.window_size,
        'mel_bins': args.mel_bins,
        'fmin': args.fmin,
        'fmax': args.fmax,
        'model_type': args.model_type,
        'recorder_number': recorder_number
    }
    all_detection_results.append({'metadata': metadata})


    # Construct the path to save the JSON file
    relative_audio_path = os.path.relpath(audio_path, start=args.output_path)
    json_filename = os.path.splitext(relative_audio_path)[0] + '.json'
    json_path = os.path.join(args.output_path, model_type, json_filename)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as json_file:
        json.dump(all_detection_results, json_file, indent=4)
    
    print(f'Saved sound event detection results to {json_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    parser_sed.add_argument('--output_path', type=str, required=True, help="Base output path to save results")
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')