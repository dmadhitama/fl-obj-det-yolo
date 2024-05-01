#!/bin/bash

# Usage:
# ./convert_labels_in_a_dir.sh <folder_path>

folder=$1

if [ -z "$folder" ]; then
  echo "Error: No folder path was provided."
  echo "Usage:./convert_labels_in_a_dir.sh <folder_path>"
  exit 1
fi

# Loop through all files in the current directory
for json_filename in $folder/*.json; do
  # Extract the filename without extension
  base_filename="${json_filename%.*}"
  # Create the new filename with .txt extension
  txt_filename="${base_filename}.txt"
  # Display the new filename
  echo "Generating $txt_filename"
  # Run the python script
  python utils/convert_labels.py --json-file $json_filename --txt-file $txt_filename
done
