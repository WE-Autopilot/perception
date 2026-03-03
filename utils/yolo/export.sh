#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model.pt>"
    exit 1
fi

MODEL_PATH="$1"

yolo export model="$MODEL_PATH" format=onnx dynamic=True simplify=True
