#!/usr/bin/env python3
"""
test_ufld_onnx_inference.py

Offline sanity check for the UFLD ONNX model.

This script:
    - Loads the ONNX model using UFldOnnx
    - Picks 5 images from a folder (e.g., Kaggle lane images)
    - Runs inference and prints output shapes

Usage example:

    python -m perception_onnx.test.test_ufld_onnx_inference \
        --onnx_path /path/to/culane_res34.onnx \
        --image_folder /path/to/kaggle_images
"""

import argparse
import glob
from pathlib import Path

import cv2

from perception_onnx.ufld_onnx_inference import UFldOnnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test UFLD ONNX inference on 5 sample images."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Folder containing test images (png/jpg/jpeg).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    image_folder = Path(args.image_folder)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_folder}")

    # Initialize UFLD ONNX wrapper
    ufld = UFldOnnx(
        engine_path=onnx_path,
        input_size=(320, 1600),
        providers=["CPUExecutionProvider"],
        use_imagenet_norm=True,
    )

    # Collect images (only png/jpg/jpeg)
    exts = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(str(image_folder / ext)))

    image_paths = sorted(image_paths)[:5]

    if not image_paths:
        raise RuntimeError(f"No images found in folder: {image_folder}")

    print(f"Running inference on {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{i}] Failed to read image: {img_path}")
            continue

        outputs = ufld.infer_raw(img)

        print(f"[{i}] Image: {img_path}")
        for j, out in enumerate(outputs):
            print(f"    Output[{j}] shape: {out.shape}")


if __name__ == "__main__":
    main()
