#!/usr/bin/env python3
"""
ufld_onnx_inference.py

ONNX Runtime inference wrapper for Ultra-Fast Lane Detection (UFLD).

This replaces the TensorRT backend with ONNX Runtime while preserving:
- Input preprocessing
- Input/output interface
- API compatibility (engine_path, config_path)

Scope:
- Inference ONLY
- No decoding or visualization
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import onnxruntime as ort


class UFldOnnx:
    """
    UFLD ONNX inference class.

    Input:
        - BGR OpenCV image
        - Shape after preprocessing: (1, 3, H, W)

    Output:
        - Raw ONNX model outputs (list of numpy arrays)
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (320, 1600),
        providers: Optional[List[str]] = None,
        use_imagenet_norm: bool = True,
    ):
        """
        Initialize ONNX Runtime session.

        Args:
            engine_path: Path to ONNX model (.onnx)
            config_path: Optional config path (kept for API parity)
            input_size: (height, width)
            providers: ONNX Runtime execution providers
            use_imagenet_norm: Apply ImageNet normalization
        """
        self.engine_path = Path(engine_path)
        self.config_path = Path(config_path) if config_path else None
        self.input_h, self.input_w = input_size
        self.use_imagenet_norm = use_imagenet_norm

        if not self.engine_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.engine_path}")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            str(self.engine_path),
            providers=providers,
        )

        # Cache I/O names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ------------------------------------------------------------------
    # Preprocessing (matches TensorRT pipeline)
    # ------------------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR image → (1, 3, H, W) float32 tensor.
        """
        img = cv2.resize(img_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.use_imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # Add batch

        return img

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer_raw(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Run ONNX inference and return raw outputs.
        """
        input_tensor = self._preprocess(img_bgr)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor},
        )

        return outputs
