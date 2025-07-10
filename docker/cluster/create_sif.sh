#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e  # Exit on any error

# Set Apptainer cache directory to be in same directory as docker_volumes
export APPTAINER_CACHEDIR="$SCRIPT_DIR/../../../apptainer_cache"

# Create cache directory if it doesn't exist
mkdir -p "$APPTAINER_CACHEDIR"

# Define the target directory relative to the script location (matching create_tar.sh)
TARGET_DIR="$SCRIPT_DIR/../../../hyak_transfer"
TAR_FILE="$TARGET_DIR/isaac-lab-base.tar"
SIF_FILE="$TARGET_DIR/isaac-lab-base.sif"

if [ ! -f "$TAR_FILE" ]; then
    echo "Error: $TAR_FILE not found at $TAR_FILE"
    exit 1
fi

# Remove existing SIF file if it exists
rm -f "$SIF_FILE"

# Build in /tmp to avoid 2GPFS permission issues (Hyak best practice)
TEMP_SIF="/tmp/$USER/isaac-lab-base.sif"
mkdir -p "/tmp/$USER"

# Convert tar archive to SIF format
apptainer build "$TEMP_SIF" "docker-archive:$TAR_FILE"

# Copy from temporary location to final destination
cp "$TEMP_SIF" "$SIF_FILE"

# Clean up temporary file
rm -f "$TEMP_SIF"

if [ $? -eq 0 ]; then
    echo "Successfully created SIF file:"
    ls -lh "$SIF_FILE"
else
    echo "Error: Failed to create SIF file"
    exit 1
fi
