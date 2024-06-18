import json
import argparse

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def compare_metadata(metadata1, metadata2):
    """Compares two metadata dictionaries to check if they are the same, ignoring the 'model_type'."""
    keys1 = set(metadata1.keys())
    keys2 = set(metadata2.keys())
    
    if keys1 != keys2:
        return False
    
    for key in keys1:
        if key != 'model_type' and metadata1[key] != metadata2[key]:
            return False
    return True

def compute_xor_nor(mask1, mask2):
    """Computes the XOR and NOR for two lists of mask values."""
    xor_result = [m1 ^ m2 for m1, m2 in zip(mask1, mask2)]
    nor_result = [m1 | m2 for m1, m2 in zip(mask1, mask2)]
    return xor_result, nor_result

def find_first_significant_discrepancy(xor_result):
    """Finds the first interval of more than 3 consecutive frames with a discrepancy."""
    count = 0
    start_index = None
    
    for i, value in enumerate(xor_result):
        if value == 1:
            count += 1
            if count == 1:
                start_index = i
            if count > 3:
                return start_index, i
        else:
            count = 0
            start_index = None
    
    return None, None

def analyze_json_files(file_path1, file_path2):
    data1 = read_json_file(file_path1)
    data2 = read_json_file(file_path2)

    # Check metadata
    metadata1 = data1[-1]['metadata']
    metadata2 = data2[-1]['metadata']
    if not compare_metadata(metadata1, metadata2):
        raise ValueError("Metadata does not match between the two files, except possibly for 'model_type'.")

    # Extract masks
    mask1 = [frame['mask'] for frame in data1[:-1]]
    mask2 = [frame['mask'] for frame in data2[:-1]]

    # Compute XOR and NOR
    xor_result, nor_result = compute_xor_nor(mask1, mask2)

    # Find the first significant discrepancy
    start, end = find_first_significant_discrepancy(xor_result)
    
    if start is not None and end is not None:
        print(f"First significant discrepancy interval: {start} to {end}")
    else:
        print("No significant discrepancy found.")
    
    print("NOR result mask:", nor_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and compare two JSON files for audio processing.")
    parser.add_argument('file_path1', type=str, help='Path to the first JSON file.')
    parser.add_argument('file_path2', type=str, help='Path to the second JSON file.')
    
    args = parser.parse_args()
    
    analyze_json_files(args.file_path1, args.file_path2)