#!/bin/bash

# Directory containing the script and frames
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMES_DIR="$SCRIPT_DIR"
OUTPUT_FILE="$FRAMES_DIR/output_video.mp4"
FRAME_RATE=30

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg is not installed. Please install it first."
    exit 1
fi

# Check if the frames directory exists
if [ ! -d "$FRAMES_DIR" ]; then
    echo "Error: Directory $FRAMES_DIR does not exist."
    exit 1
fi

echo "Creating video from frames in $FRAMES_DIR..."

# Run ffmpeg
# -y: overwrite output file
# -r: frame rate
# -i: input pattern (e.g., frame0000.png)
# -c:v: video codec (libx264)
# -pix_fmt: pixel format (yuv420p for compatibility)
# -vf: video filters (pad width/height to be even numbers for H.264)
ffmpeg -y -r $FRAME_RATE -i "$FRAMES_DIR/frame%04d.png" \
    -c:v libx264 -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Video successfully created: $OUTPUT_FILE"
else
    echo "Error: Failed to create video."
fi
