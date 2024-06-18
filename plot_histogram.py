import json
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np

# Define la lista de etiquetas de interés
labels = ["Speech", "Singing", "Male singing", "Female singing", "Child singing", 
          "Male speech, man speaking", "Female speech, woman speaking", "Conversation", 
          "Narration, monologue", "Music"]

def load_data(json_path):
    """Load thresholds, average percentages, and total JSON file count from a JSON file."""
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Aseguramos que accedemos correctamente a los datos
        thresholds = data['thresholds']
        average_percentages = data['average_percentages']
        total_json_files = data['total_json_files']

        return thresholds, average_percentages, total_json_files
    except KeyError as e:
        print(f"Error: La clave {e} no se encuentra en el archivo {json_path}")
        return None, None, 0
    except json.JSONDecodeError as e:
        print(f"Error: El archivo {json_path} no está bien formateado como JSON: {e}")
        return None, None, 0
    except Exception as e:
        print(f"Error no esperado al cargar {json_path}: {e}")
        return None, None, 0


def plot_results(thresholds, weighted_percentages, output_path):
    """Plot the graph of thresholds vs. weighted average percentages and save it as an image."""
    if thresholds is None or len(thresholds) == 0 or weighted_percentages is None or len(weighted_percentages) == 0:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, weighted_percentages, marker='o', linestyle='-', color='b')
    plt.title('Threshold Impact on Label Confidence')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Percentage Above Threshold (%)')
    plt.grid(True)
    
    plt.xlim([0, 1])
    plt.ylim([0, 35])

    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")


def process_directory(directory, output_file):
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found.")
        return

    total_counts = np.array([])
    weighted_sums = np.array([])

    for json_file in json_files:
        thresholds, percentages, num_files = load_data(json_file)
        if percentages is None:
            continue

        if total_counts.size == 0:
            total_counts = np.zeros(len(percentages))
            weighted_sums = np.zeros(len(percentages))
        
        total_counts += num_files
        weighted_sums += np.array(percentages) * num_files
    
    weighted_percentages = weighted_sums / total_counts if total_counts.size != 0 else []

    plot_results(thresholds, weighted_percentages, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot annotation analysis for multiple audio labels from JSON files in a directory.")
    parser.add_argument('directory', type=str, help='Directory containing JSON files.')
    parser.add_argument('output_file', type=str, help='Path to save the output plot image.')
    
    args = parser.parse_args()
    
    process_directory(args.directory, args.output_file)




