#!/usr/bin/env python3
"""
ufld_onnx_inference.py

ONNX inference wrapper for Ultra-Fast Lane Detection (UFLD).

This class:
    - Loads an ONNX lane detection model with ONNX Runtime
    - Handles image preprocessing (resize, normalize, NCHW)
    - Runs inference and returns raw outputs

It is designed to mirror a typical "UFLD" class used in TensorRT-based
pipelines, but uses ONNX Runtime instead.

Model expectations:
    - Input shape: (1, 3, H, W) = (1, 3, 320, 1600) by default
    - Input format: float32, normalized image (0–1 or ImageNet)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


class UFldOnnx:
    """
    UFLD ONNX inference class.

    Typical usage:
        ufld = UFldOnnx(engine_path="models/culane_res34.onnx")
        outputs = ufld.infer_raw(bgr_image)
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (320, 1600),
        providers: Optional[list] = None,
        use_imagenet_norm: bool = True,
    ):
        """
        Initialize ONNX Runtime session for UFLD model.

        Args:
            engine_path: Path to the ONNX model file.
            config_path: Optional path to extra config (kept for API parity
                         with TensorRT UFLD implementations).
            input_size:  (height, width) expected by the model.
            providers:   ONNX Runtime providers, e.g. ["CPUExecutionProvider"]
                         or ["CUDAExecutionProvider", "CPUExecutionProvider"].
            use_imagenet_norm: If True, apply ImageNet mean/std after scaling.
        """
        self.engine_path = str(engine_path)
        self.config_path = str(config_path) if config_path is not None else None
        self.input_h, self.input_w = input_size
        self.use_imagenet_norm = use_imagenet_norm

        # Default providers: GPU + CPU if available, otherwise CPU.
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Create an ONNX Runtime inference session.
        self.session = ort.InferenceSession(self.engine_path, providers=providers)

        # Cache input/output names to avoid repeated lookups.
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image into a batched NCHW tensor ready for the model.

        Steps:
            1. Resize to (input_w, input_h)
            2. Convert BGR -> RGB
            3. Normalize to [0, 1]
            4. (Optionally) apply ImageNet mean/std normalization
            5. Convert HWC -> CHW
            6. Add batch dimension

        Args:
            img_bgr: Input image in OpenCV BGR format (uint8).

        Returns:
            Numpy array of shape (1, 3, input_h, input_w), dtype float32.
        """
        # Resize to the network input size (width, height).
        img = cv2.resize(img_bgr, (self.input_w, self.input_h))

        # Convert from BGR (OpenCV default) to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale to [0, 1].
        img = img.astype(np.float32) / 255.0

        # Optionally apply ImageNet normalization.
        if self.use_imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

        # HWC -> CHW.
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension -> (1, 3, H, W).
        img = np.expand_dims(img, axis=0)

        return img

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    def infer_raw(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Run the ONNX model and return the raw outputs.

        Args:
            img_bgr: Input OpenCV BGR image.

        Returns:
            List of numpy arrays corresponding to ONNX model outputs.
        """
        # Prepare input tensor.
        input_tensor = self._preprocess(img_bgr)

        # Run inference using ONNX Runtime.
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor},
        )

        return outputs

    # -------------------------------------------------------------------------
    # OPTIONAL: HIGH-LEVEL LANE DECODING (STUB)
    # -------------------------------------------------------------------------
    def infer_lanes(self, img_bgr: np.ndarray):
        """
        High-level API to run inference and decode lane lines.

        NOTE:
            The actual decoding logic depends on how the model was trained
            and on the original TensorRT implementation. This method is
            provided as a placeholder for future extension.

        Args:
            img_bgr: Input OpenCV BGR image.

        Returns:
            Currently returns the raw outputs from the model.
            In the future, this could be a list of lane polylines or points.
        """
        outputs = self.infer_raw(img_bgr)
        # TODO: Implement real lane post-processing here using the
        #       original TensorRT/UFLD decoding logic.
        return outputs
