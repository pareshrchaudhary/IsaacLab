#!/bin/bash

# Isaac Sim Asset Setup Script
# This script downloads and extracts Isaac Sim assets to the correct location for Docker usage
set -e  # Exit on any error

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
ASSETS_DIR="$PROJECT_ROOT/docker_volumes/assets"

echo "Project root: $PROJECT_ROOT"
echo "Assets target directory: $ASSETS_DIR"
echo

# Check if assets already exist in the correct nested structure
ISAAC_ASSETS_PATH="$ASSETS_DIR/Assets/Isaac/4.5"
if [ -d "$ISAAC_ASSETS_PATH/NVIDIA" ] && [ -d "$ISAAC_ASSETS_PATH/Isaac" ]; then
    echo "Isaac Sim assets already exist at: $ISAAC_ASSETS_PATH"
    echo "You can proceed with running Isaac Sim."
    exit 0
fi

echo "Isaac Sim assets not found at: $ISAAC_ASSETS_PATH"
echo

# Create temporary directory for downloads
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Cleanup function to remove temp directory on exit
cleanup() {
    echo "Cleaning up temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Asset download URLs and filenames
declare -a ASSET_URLS=(
    "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-1%404.5.0-rc.36%2Brelease.19112.f59b3005.zip"
    "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-2%404.5.0-rc.36%2Brelease.19112.f59b3005.zip"
    "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-3%404.5.0-rc.36%2Brelease.19112.f59b3005.zip"
)

declare -a ASSET_FILES=(
    "isaac-sim-assets-1@4.5.0-rc.36+release.19112.f59b3005.zip"
    "isaac-sim-assets-2@4.5.0-rc.36+release.19112.f59b3005.zip"
    "isaac-sim-assets-3@4.5.0-rc.36+release.19112.f59b3005.zip"
)

# Download asset files
echo "Downloading Isaac Sim asset files..."

for i in "${!ASSET_URLS[@]}"; do
    url="${ASSET_URLS[$i]}"
    filename="${ASSET_FILES[$i]}"
    filepath="$TEMP_DIR/$filename"
    current=$((i + 1))
    total=${#ASSET_URLS[@]}
    
    echo "[$current/$total] Downloading: $filename"
    
    # Download with progress bar
    if ! wget --progress=bar:force:noscroll -O "$filepath" "$url"; then
        echo "Error: Failed to download $filename"
        exit 1
    fi
    
    echo "Downloaded: $filename"
done

echo "All asset files downloaded successfully"
echo

# Create the target directory structure
echo "Creating target directory structure..."
mkdir -p "$ASSETS_DIR"
echo "Created: $ASSETS_DIR"
echo

# Extract assets
echo "Extracting Isaac Sim asset packs..."

for i in "${!ASSET_FILES[@]}"; do
    filename="${ASSET_FILES[$i]}"
    filepath="$TEMP_DIR/$filename"
    current=$((i + 1))
    total=${#ASSET_FILES[@]}
    
    echo "[$current/$total] Extracting: $filename"
    
    # Check if file exists
    if [ ! -f "$filepath" ]; then
        echo "Error: Asset file not found: $filepath"
        exit 1
    fi
    
    # Get file size for progress indication
    file_size=$(du -h "$filepath" | cut -f1)
    echo "File size: $file_size"
    
    # Extract with simple progress indication
    echo "Extracting... (this may take several minutes)"
    if ! unzip -q "$filepath" -d "$ASSETS_DIR"; then
        echo "Error: Failed to extract $filename"
        exit 1
    fi
    
    echo "Completed extraction of $filename"
done

echo "Asset extraction complete"
echo

# Ensure proper nested structure exists
echo "Ensuring proper asset directory structure..."
mkdir -p "$ISAAC_ASSETS_PATH"

# Check if assets were extracted to the root level and need to be moved
if [ -d "$ASSETS_DIR/NVIDIA" ] && [ -d "$ASSETS_DIR/Isaac" ] && [ ! -d "$ISAAC_ASSETS_PATH/NVIDIA" ]; then
    echo "Moving assets to proper nested structure..."
    
    # Move NVIDIA folder
    if [ -d "$ASSETS_DIR/NVIDIA" ]; then
        mv "$ASSETS_DIR/NVIDIA" "$ISAAC_ASSETS_PATH/" || {
            echo "Error: Failed to move NVIDIA folder"
            exit 1
        }
    fi
    
    # Move Isaac folder  
    if [ -d "$ASSETS_DIR/Isaac" ]; then
        mv "$ASSETS_DIR/Isaac" "$ISAAC_ASSETS_PATH/" || {
            echo "Error: Failed to move Isaac folder"
            exit 1
        }
    fi
    
    # Move any other files that might be at root level
    find "$ASSETS_DIR" -maxdepth 1 -type f -exec mv {} "$ISAAC_ASSETS_PATH/" \; 2>/dev/null || true
    
    echo "Assets moved to: $ISAAC_ASSETS_PATH"
fi
echo

# Verify the extraction
echo "Verifying asset structure..."
if [ -d "$ISAAC_ASSETS_PATH/NVIDIA" ] && [ -d "$ISAAC_ASSETS_PATH/Isaac" ]; then
    nvidia_count=$(find "$ISAAC_ASSETS_PATH/NVIDIA" -type f 2>/dev/null | wc -l)
    isaac_count=$(find "$ISAAC_ASSETS_PATH/Isaac" -type f 2>/dev/null | wc -l)
    
    echo "Assets extracted successfully!"
    echo "- NVIDIA folder: $nvidia_count files"
    echo "- Isaac folder: $isaac_count files"
    echo
    echo "Isaac Sim assets are ready for use!"
    echo "Location: $ISAAC_ASSETS_PATH"
else
    echo "Error: Asset extraction failed - missing required folders"
    echo "Expected: $ISAAC_ASSETS_PATH/NVIDIA and $ISAAC_ASSETS_PATH/Isaac"
    
    if [ -d "$ASSETS_DIR" ]; then
        echo "Found in assets directory:"
        ls -la "$ASSETS_DIR" 2>/dev/null || echo "Directory listing failed"
        echo
        echo "Directory structure:"
        find "$ASSETS_DIR" -maxdepth 3 -type d 2>/dev/null | head -20
    else
        echo "Assets directory does not exist"
    fi
    
    exit 1
fi

echo
echo "Setup completed successfully!"