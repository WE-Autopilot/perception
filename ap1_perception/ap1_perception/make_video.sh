#!/bin/bash

# Check if a directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_frames_directory>"
    echo "Example: $0 yolo_frames"
    exit 1
fi

FRAMES_DIR="$1"
# Resolve to absolute path to avoid confusion
FRAMES_DIR_ABS=$(cd "$FRAMES_DIR" && pwd)
OUTPUT_FILE="$FRAMES_DIR_ABS/output_video.mp4"
FRAME_RATE=30

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    exit 1
fi

# Check if the directory exists
if [ ! -d "$FRAMES_DIR_ABS" ]; then
    echo "Error: Directory $FRAMES_DIR does not exist."
    exit 1
fi

# Count PNG frames to see if there's anything to process
FRAME_COUNT=$(ls -1 "$FRAMES_DIR_ABS"/frame*.png 2>/dev/null | wc -l)
if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "Error: No frames matching 'frame*.png' found in $FRAMES_DIR_ABS"
    exit 1
fi

echo "Found $FRAME_COUNT frames. Creating 30 FPS video in $FRAMES_DIR_ABS..."

# Run ffmpeg
ffmpeg -y -r $FRAME_RATE -i "$FRAMES_DIR_ABS/frame%04d.png" \
    -c:v libx264 -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Video successfully created: $OUTPUT_FILE"
    echo "------------------------------------------------"
else
    echo "Error: Failed to create video."
    exit 1
fi
