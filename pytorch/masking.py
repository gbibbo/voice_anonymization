import json
import argparse
import os
import numpy as np

def read_json_and_get_indices(json_path, label='Speech', threshold=0.4):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list) or 'metadata' not in data[-1]:
        print(f"Unexpected JSON structure in {json_path}")
        return [], [], 0, 0
    
    metadata = data[-1]['metadata']
    hop_size = metadata.get('hop_size', 0)
    window_size = metadata.get('window_size', 0)
    detection_results = data[:-1]

    speech_frames = []
    frame_mask = [1] * len(detection_results)

    for i, entry in enumerate(detection_results):
        frame_contains_speech = False
        predictions = entry.get('predictions', [])
        for prediction in predictions:
            if prediction.get('class') == label and prediction.get('prob', 0) > threshold:
                frame_contains_speech = True
                break
        if frame_contains_speech:
            speech_frames.append(entry.get('frame_index', 0) * hop_size)
            frame_mask[i] = 0

    return speech_frames, frame_mask, hop_size, window_size

def create_mask_json(original_json_path, mask, output_json_path, threshold):
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    try:
        with open(original_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {original_json_path}: {e}")
        return

    detection_results = data[:-1]
    metadata = data[-1]  # This is the last item which should be the metadata

    # Add the 'threshold' to the metadata
    metadata['metadata']['threshold'] = threshold

    new_detection_results = []

    for i, entry in enumerate(detection_results):
        new_entry = {
            'frame_index': entry.get('frame_index'),
            'sample_index': entry.get('sample_index'),
            'time': entry.get('time'),
            'mask': mask[i]
        }
        new_detection_results.append(new_entry)
    
    # Combine the modified detection results with the updated metadata
    final_data = new_detection_results + [metadata]

    with open(output_json_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Mask JSON saved in: {output_json_path}")

def main(json_filename, threshold, output_folder):
    json_path = json_filename
    
    speech_frames, frame_mask, hop_size, window_size = read_json_and_get_indices(json_path, threshold=threshold)
    
    output_json_path = os.path.join(output_folder, os.path.basename(json_path).replace('.json', '_mask.json'))
    print('output_folder = ', output_folder)
    print('os.path.basename(json_path).replace(.json, _mask.json) = ', os.path.basename(json_path).replace('.json', '_mask.json'))
    create_mask_json(json_path, frame_mask, output_json_path, threshold=threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a JSON mask to mark 'Speech' sections.")
    parser.add_argument('json_filename', type=str, help='Input JSON file name with detection results.')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Probability threshold to determine speech presence.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Folder where the mask JSON file will be saved.')
    
    args = parser.parse_args()
    
    main(args.json_filename, args.threshold, args.output_folder)
