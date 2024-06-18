#!/bin/bash

# Crear y activar un entorno virtual con Python 3.10
source /mnt/fast/nobackup/users/gb0048/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/fast/nobackup/users/gb0048/miniconda3/envs/myenv

# Navigate to the main directory containing the script and audio folders
#cd /mnt/c/Users/bibbo/Downloads/audioset_tagging_cnn-master
cd /mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master || exit

# Path to the 'tests' directory where subdirectories are located
#BASE_DIR="resources/tests"
BASE_DIR="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Audios"

# Echo the base directory to debug
echo "Base directory: $BASE_DIR"

# Path to the checkpoint file
#CHECKPOINT_PATH="/mnt/c/Users/bibbo/Downloads/audioset_tagging_cnn-master/Cnn14_DecisionLevelMax_mAP=0.385.pth"
#CHECKPOINT_PATH="/mnt/c/Users/bibbo/Downloads/audioset_tagging_cnn-master/Cnn14_DecisionLevelAtt_mAP=0.425"
CHECKPOINT_PATH="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/Cnn14_DecisionLevelAtt_mAP=0.425"
#CHECKPOINT_PATH="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/Cnn14_DecisionLevelMax_mAP=0.385"

# Check if the checkpoint file exists; if not, download it
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading checkpoint..."
    wget -O "$CHECKPOINT_PATH" https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1
    #wget -O "$CHECKPOINT_PATH" https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth?download=1
fi

# Model type
#MODEL_TYPE="Cnn14_DecisionLevelMax"
MODEL_TYPE="Cnn14_DecisionLevelAtt"

# Calculate the total number of WAV files to be processed
total_files=$(find "$BASE_DIR" -type f -name "*.WAV" | wc -l)
echo "Total .WAV files to process: $total_files"

# Calculate the estimated time to process all files
estimated_time=$((total_files * 45))  # 45 seconds per file on interactive session in Condor
echo "Estimated total processing time: $((estimated_time / 60)) minutes"

# Counter for processed files
processed_files=0

# Directory to save output results - make sure this exists
OUTPUT_DIR="$BASE_DIR/Cnn14_DecisionLevelMax"
mkdir -p "$OUTPUT_DIR"

# Iterate over each subdirectory in the 'tests' directory
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"

        # Iterate over each audio file in the directory
        for audio_file in "$dir"/*.WAV; do
            if [ -f "$audio_file" ]; then
                echo "  Processing audio file: $audio_file"

                # Extract the filename without the path and extension
                filename=$(basename -- "$audio_file")
                filename="${filename%.*}"

                # Define the path for the output JSON file
                json_output_path="$dir/$filename.json"

                # Run the sound event detection script
                CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py sound_event_detection \
                    --model_type=$MODEL_TYPE \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --audio_path="$audio_file" \
                    --cuda --sample_rate=48000 \
                    --output_path="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths"
                    #--output_path="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/resources" 

                # Increment the count of processed files
                ((processed_files++))

                # Calculate and print the progress and estimated remaining time
                progress=$(echo "$processed_files * 100 / $total_files" | bc)
                remaining_time=$(( (total_files - processed_files) * 45 ))
                echo "Progress: $progress% - Estimated time remaining: $((remaining_time / 60)) minutes"
            else
                echo "No audio files found in $dir"
            fi
        done
    else
        echo "No directories found in $BASE_DIR"
    fi
done

echo "All directories have been processed."
