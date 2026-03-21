#!/bin/bash

# MaxLab Library Setup Helper
# This script extracts the MaxLab library from the zip file

set -e

echo "MaxLab Library Setup"
echo "===================="
echo ""

# Find zip file
MAXLAB_ZIP=$(ls "$HOME/MaxLab/share/libmaxlab-"*.zip 2>/dev/null | head -n 1)

if [ -z "$MAXLAB_ZIP" ]; then
    echo "ERROR: No MaxLab zip file found in ~/MaxLab/share/"
    echo "Expected file like: libmaxlab-1.0.0_*.zip"
    exit 1
fi

echo "Found: $MAXLAB_ZIP"

# Check if already extracted
if [ -d "$HOME/MaxLab/share/maxlab_lib" ]; then
    echo ""
    echo "WARNING: maxlab_lib directory already exists!"
    read -p "Re-extract anyway? [y/N]: " REEXTRACT
    if [[ ! "$REEXTRACT" =~ ^[Yy]$ ]]; then
        echo "Keeping existing installation."
        echo "Location: $HOME/MaxLab/share/maxlab_lib"
        exit 0
    fi
    rm -rf "$HOME/MaxLab/share/maxlab_lib"
fi

# Extract
echo ""
echo "Extracting MaxLab library..."
unzip -q "$MAXLAB_ZIP" -d "$HOME/MaxLab/share/"

# Verify extraction
if [ -d "$HOME/MaxLab/share/maxlab_lib" ]; then
    echo ""
    echo "SUCCESS! MaxLab library extracted to:"
    echo "  $HOME/MaxLab/share/maxlab_lib"
    echo ""
    echo "Contents:"
    ls -la "$HOME/MaxLab/share/maxlab_lib/"
    echo ""
    echo "You can now build the closed-loop experiment:"
    echo "  cd $(dirname $0)"
    echo "  ./quickstart.sh"
else
    echo ""
    echo "ERROR: Extraction completed but maxlab_lib directory not found!"
    echo "The zip file structure may be different than expected."
    echo ""
    echo "Extracted contents:"
    ls -la "$HOME/MaxLab/share/"
    exit 1
fi
