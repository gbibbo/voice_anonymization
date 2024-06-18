#!/bin/bash

# Crear y activar un entorno virtual con Python 3.10
source /mnt/fast/nobackup/users/gb0048/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/fast/nobackup/users/gb0048/miniconda3/envs/myenv

# Navigate to the main directory containing the script and audio folders
cd /mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master || exit

# Base directory where the original audio files are located
BASE_AUDIO_DIR="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Audios"

# Base directory for DecisionLevelAtt labels
JSON_DIR_ATT="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Cnn14_DecisionLevelAtt"

# Base directory for DecisionLevelMax labels
JSON_DIR_MAX="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Cnn14_DecisionLevelMax"

# Loop through all subdirectories in the audio directory
for dir in $BASE_AUDIO_DIR/*; do
    # Create a new directory for the modified files
    MODIFIED_DIR="${BASE_AUDIO_DIR}_modified/$(basename $dir)"
    mkdir -p "$MODIFIED_DIR"

    # Loop through each WAV file in the subdirectories
    for wav_file in $dir/*.WAV; do
        # Extract the base name of the file without the extension
        base_name=$(basename "$wav_file" .WAV)

        # Paths to the corresponding JSON files
        json_file_att="$JSON_DIR_ATT/${dir##*/}/$base_name.json"
        json_file_max="$JSON_DIR_MAX/${dir##*/}/$base_name.json"

        # Output paths for the modified files
        output_att="$MODIFIED_DIR/${base_name}_modified2.WAV"
        output_max="$MODIFIED_DIR/${base_name}_modified.WAV"

        # Run the cleaning script with DecisionLevelAtt labels
        echo "Processing $wav_file with ATT labels..."
        python3 pytorch/cleaning.py "$json_file_att" "$wav_file" "$output_att"

        # Run the cleaning script with DecisionLevelMax labels
        echo "Processing $output_att with MAX labels..."
        python3 pytorch/cleaning.py "$json_file_max" "$output_att" "$output_max"

        # Delete the intermediate modified file
        echo "Deleting intermediate file $output_att..."
        rm "$output_att"
    done
done

echo "Processing completed for all files."


