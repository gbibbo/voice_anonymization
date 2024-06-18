import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Define la lista de etiquetas de interés
labels = ["Speech", "Singing", "Male singing", "Female singing", "Child singing", 
          "Male speech, man speaking", "Female speech, woman speaking", "Conversation", 
          "Narration, monologue", "Music"]

def process_file(filepath):
    """Process a single JSON file and return the maximum probabilities for defined labels, including zeros for non-matching."""
    with open(filepath, 'r') as file:
        data = json.load(file)

    probabilities = []
    frame_index = 0  # Iniciar contador de frames
    for frame in data[:-1]:  # Excluye metadatos al final
        if 'predictions' in frame and frame['predictions']:
            # Filtrar solo las predicciones que están en la lista de etiquetas de interés
            filtered_predictions = [pred for pred in frame['predictions'] if pred['class'] in labels]
            # Buscar la probabilidad máxima entre las etiquetas filtradas, asigna 0 si ninguna está presente
            if filtered_predictions:
                max_pred = max(filtered_predictions, key=lambda pred: pred['prob'])
                max_prob = max_pred['prob']
                label = max_pred['class']
            else:
                max_prob = 0
                label = 'None'
            probabilities.append(max_prob)
            #print(f"Frame {frame_index}: Max Label = {label}, Max Prob = {max_prob}")
        else:
            probabilities.append(0)
            print(f"Frame {frame_index}: No predictions available or no matching labels found.")

        frame_index += 1  # Incrementar el contador de frames

    return probabilities



def analyze_files(directory):
    """Analyze all JSON files in each subdirectory and compute the average percentages."""
    thresholds = np.linspace(0, 1, num=100)
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for subdirectory in subdirectories:
        cumulative_percentages = np.zeros_like(thresholds)
        json_files = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory) if f.endswith('.json')]
        total_files = len(json_files)
        
        if total_files == 0:
            continue  # Skip if no JSON files are found
        
        total_frames = 0

        for filepath in json_files:
            probabilities = process_file(filepath)
            total_frames += len(probabilities)
            
            # Update counts for each threshold
            for i, threshold in enumerate(thresholds):
                cumulative_percentages[i] += np.sum(np.array(probabilities) > threshold)
        
        # Average the percentages across all files
        print('cumulative_percentages = ', cumulative_percentages)
        average_percentages = (cumulative_percentages / total_frames) * 100
        
        # Prepare data for output
        output_data = {
            "thresholds": thresholds.tolist(),
            "average_percentages": average_percentages.tolist(),
            "total_json_files": total_files
        }
        
        # Save the plot data for this subdirectory to a JSON file
        subdirectory_name = os.path.basename(subdirectory)
        output_filename = os.path.join(directory, f"{subdirectory_name}.json")
        with open(output_filename, 'w') as out_file:
            json.dump(output_data, out_file, indent=4)

        print(f"Data saved to {output_filename}")

if __name__ == '__main__':
    directory = "/mnt/d/Cnn14_DecisionLevelAtt"
    analyze_files(directory)

