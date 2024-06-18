#!/bin/bash

# Crear y activar un entorno virtual con Python 3.10
source /mnt/fast/nobackup/users/gb0048/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/fast/nobackup/users/gb0048/miniconda3/envs/myenv

# Navigate to the main directory containing the script and audio folders
cd /mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master || exit

# Define the base directory where the original JSON annotations are stored
BASE_DIR="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Cnn14_DecisionLevelMax"
#BASE_DIR="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/resources/Cnn14_DecisionLevelAtt"

# Define the destination directory where the masks should be saved
DEST_DIR="/mnt/fast/nobackup/scratch4weeks/gb0048/20231128_SWILL_AudioMoths/Cnn14_DecisionLevelMax_MASKS"
#DEST_DIR="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/resources/Cnn14_DecisionLevelAtt_MASKS"

# Define the threshold for the masking process (adjust as needed)
THRESHOLD=0.4

# Path to the directory where masking.py is located
SCRIPT_DIR="/mnt/fast/nobackup/users/gb0048/audioset_tagging_cnn-master/pytorch"

# Check if the DEST_DIR exists, if not, create it
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory $DEST_DIR does not exist. Creating it..."
    mkdir -p "$DEST_DIR"
fi

# Loop over all subdirectories within the base directory
for SUBDIR in $(ls $BASE_DIR); do
    # Check if the subdirectory exists in the destination directory, if not create it
    if [ ! -d "$DEST_DIR/$SUBDIR" ]; then
        echo "Creating subdirectory $DEST_DIR/$SUBDIR"
        mkdir -p "$DEST_DIR/$SUBDIR"
    fi

    # Loop over all .json files in the subdirectory
    for JSON_FILE in "$BASE_DIR/$SUBDIR"/*.json; do
        # Skip any already masked files
        if [ "${JSON_FILE}" != "${JSON_FILE%_mask.json}" ]; then
            echo "Skipping already processed file $JSON_FILE"
            continue
        fi

        # Extract the filename without the extension
        FILENAME=$(basename "$JSON_FILE" .json)

        # Define the path for the output mask JSON file
        OUTPUT_JSON="$DEST_DIR/$SUBDIR"

        # Run the masking.py script to generate the mask and save it to the destination directory
        echo "Processing $JSON_FILE -> $OUTPUT_JSON"
        python3 "$SCRIPT_DIR/masking.py" -o "$OUTPUT_JSON" -t $THRESHOLD "$JSON_FILE"
    done
done

echo "All JSON files have been processed and corresponding masks have been saved."