#!/bin/bash

# This script downloads the zip file containing inference.py and the models from Google Drive and extracts it.

# Function to check if a command exists
command_exists () {
    command -v "$1" &> /dev/null ;
}

# Install gdown if not already installed
if ! command_exists gdown ; then
    echo "gdown could not be found. Installing gdown..."
    python3 -m pip install gdown
fi

# Replace 'YOUR_ZIP_FILE_ID' with the actual file ID from Google Drive
# https://drive.google.com/file/d/1WfmQSFIoYR_VoQmTauvQy6dAJVpV-K8a/view?usp=sharing
ZIP_FILE_ID=1WfmQSFIoYR_VoQmTauvQy6dAJVpV-K8a

# Download the zip file
echo "Downloading the zip file containing inference.py and models..."
python3 -m gdown --id $ZIP_FILE_ID -O finetuned_mt5.zip

# Unzip the file
echo "Extracting the zip file..."
unzip finetuned_mt5.zip -d finetuned_mt5
rm finetuned_mt5.zip

echo "Download and extraction complete."