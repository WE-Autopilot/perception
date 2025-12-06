from pathlib import Path
from typing import Optional, Union, Sequence
from ultralytics import YOLO
from PIL import Image


class StopSignDetectionModel:
	"""Encapsulates training, validation, export, and inference for the Stop Sign detector.

	This class mirrors and organizes the functionality from the Jupyter notebook into a reusable module.
	"""

    # initializer
	def __init__(self, data_yaml: Union[str, Path] = "Stop-Sign-3/data.yaml", device: str = "cpu") -> None:
		self.data_yaml = str(data_yaml)
		self.device = device

	# ---------------------
	# Dataset utilities
	# ---------------------
	
    # Show data.yaml content
	def show_data_yaml(self, path: Optional[Union[str, Path]] = None) -> None:
		p = Path(path) if path is not None else Path(self.data_yaml)
		if p.exists():
			print(p.read_text())
		else:
			print(f"{p} not found; check the path")

    # Check train set consistency
	def check_train_consistency(self, train_root: Union[str, Path] = "Stop-Sign-3/train") -> None:
		train_root = Path(train_root)
		imgs_dir = train_root / "images"
		labels_dir = train_root / "labels"

		imgs = list(imgs_dir.glob("*")) if imgs_dir.exists() else []
		labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

		print("images:", len(imgs))
		print("labels:", len(labels))

		img_basenames = {p.stem for p in imgs}
		label_basenames = {p.stem for p in labels}

		missing_labels = sorted(img_basenames - label_basenames)
		extra_labels = sorted(label_basenames - img_basenames)

		print("missing label files for images:", len(missing_labels))
		print("label files with no image:", len(extra_labels))
		if missing_labels:
			print("Examples of images missing labels:", missing_labels[:10])

    # Preview label files
	def preview_label_files(self, labels_dir: Union[str, Path] = "Stop-Sign-3/train/labels", limit: int = 5) -> None:
		labels_dir = Path(labels_dir)
		
		if labels_dir.exists():
			files = sorted(labels_dir.glob("*.txt"))[:limit]
			if not files:
				print("No .txt label files found in", labels_dir)
			for p in files:
				print("---", p.name, "---")
				print(p.read_text())
		else:
			print(f"{labels_dir} not found; check the path")

	# ---------------------
	# Weights helpers
	# ---------------------
	
    # Get Stage1 weights path
	@staticmethod
	def get_stage1_weights() -> str:
		p_best = Path("runs/detect/finetune_stage1/weights/best.pt")
		p_last = Path("runs/detect/finetune_stage1/weights/last.pt")
		if p_best.exists():
			return str(p_best)
		if p_last.exists():
			return str(p_last)
		raise FileNotFoundError("No Stage1 weights found in runs/detect/finetune_stage1/weights")

    # Load model from weights
	@staticmethod
	def load_model(weights: Union[str, Path]) -> YOLO:
		return YOLO(str(weights))

	# ---------------------
	# Validation & export
	# ---------------------
	
    # Validate model
	def validate(self, weights: Union[str, Path]) -> object:
		m = self.load_model(weights)
		print("Validating", weights)
		return m.val(data=self.data_yaml, device=self.device)

    # Export to ONNX
	def export_onnx(self, weights: Union[str, Path], out_dir: Union[str, Path] = "yolo_models/", imgsz: int = 640) -> Optional[Path]:
		m = self.load_model(weights)
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)
		print("Exporting", weights, "to ONNX (imgsz=", imgsz, ")...")
		m.export(format="onnx", imgsz=imgsz, device=self.device)
		# Try to copy produced ONNX next to weights into out_dir
		weights_path = Path(str(weights))
		candidate = weights_path.with_suffix(".onnx")
		if candidate.exists():
			target = out_dir / candidate.name
			try:
				import shutil
				shutil.copy(candidate, target)
				print("Copied", candidate, "->", target)
				return target
			except Exception as e:
				print("Copy failed:", e)
		else:
			print("No ONNX file found next to", weights_path, "; check exporter output.")
		return None

	# ---------------------
	# Training
	# ---------------------
	
    # Train Stage 1
	def train_stage1(
		self,
		base_weights: Union[str, Path] = "yolo11n.pt",
		epochs: int = 20,
		imgsz: int = 320,
		batch: int = 2,
		name: str = "finetune_stage1",
	) -> None:
		print("Stage1: base_weights=", base_weights)
		m = YOLO(str(base_weights))
		# Minimal override keeps model path without complex cfg alignment
		m.overrides = getattr(m, "overrides", {}) or {}
		m.overrides["model"] = str(base_weights)
		m.train(
			data=self.data_yaml,
			epochs=epochs,
			imgsz=imgsz,
			batch=batch,
			device=self.device,
			freeze=[0],
			augment=True,
			name=name,
		)
		print("Stage1 completed, outputs in runs/detect/", name)

    # Train Stage 2
	def train_stage2(
		self,
		start_weights: Union[str, Path],
		epochs: int = 30,
		imgsz: int = 640,
		batch: int = 2,
		name: str = "finetune_stage2",
		auto_augment: str = "randaugment",
		mixup: float = 0.05,
		copy_paste: float = 0.2,
	) -> None:
		print("Stage2: starting from", start_weights)
		m = YOLO(str(start_weights))
		m.overrides = getattr(m, "overrides", {}) or {}
		m.overrides["model"] = str(start_weights)
		m.train(
			data=self.data_yaml,
			epochs=epochs,
			imgsz=imgsz,
			batch=batch,
			device=self.device,
			augment=True,
			auto_augment=auto_augment,
			mixup=mixup,
			copy_paste=copy_paste,
			name=name,
		)
		print("Stage2 completed, outputs in runs/detect/", name)

    # Continue training from Stage1 weights
	def continue_from_stage1(
		self,
		epochs: int = 30,
		imgsz: int = 640,
		batch: int = 2,
	) -> None:
		try:
			w = self.get_stage1_weights()
		except Exception as e:
			print("Stage1 weights not found:", e)
			return
		self.train_stage2(w, epochs=epochs, imgsz=imgsz, batch=batch)

	# ---------------------
	# Inference helpers
	# ---------------------
	
    # Predict on image
	def predict_image(
		self,
		weights: Union[str, Path],
		onnx_model_path: Optional[Union[str, Path]],
		image_path: Union[str, Path],
		imgsz: int = 640,
		conf: float = 0.25,
		iou: float = 0.45,
		save_dir: Optional[Union[str, Path]] = None,
	) -> Sequence:
		"""Predict on a single image using Ultralytics runtime (PT or ONNX). Returns list of Results."""
		print("The model will use", "ONNX" if onnx_model_path is not None else "PT", "weights for inference.")
		m = YOLO(str(onnx_model_path) if onnx_model_path is not None else str(weights))
		kwargs = dict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, device=self.device)
		if save_dir is not None:
			# Some versions support `project`/`name` or `save=True`. We use save=True for simplicity.
			kwargs["save"] = True
		results = m.predict(**kwargs)
		if save_dir is not None:
			out_path = Path(save_dir)
			out_path.mkdir(parents=True, exist_ok=True)
			for i, r in enumerate(results):
				annotated_img = r.plot()
				save_path = out_path / f"result_{i}.jpg"
				Image.fromarray(annotated_img).save(save_path)
			print(f"Results saved to {out_path}")

		return results



if __name__ == "__main__":
    # Simple test code
    model = StopSignDetectionModel(device="cpu")

    # check dataset
    model.show_data_yaml(path="Stop-Sign-3/data.yaml")
    model.check_train_consistency()
    model.preview_label_files()
	
    # train stage 1
    # model.train_stage1(base_weights="yolo11n.pt", epochs=20, imgsz=320, batch=2)
	
    # train stage 2
    # model.train_stage2(start_weights="runs/detect/finetune_stage1/weights/best.pt", epochs=50, imgsz=640, batch=2, device="cpu")
	
    # test
    model.predict_image(
		weights="runs/detect/finetune_stage2/weights/best.pt",
		onnx_model_path="yolo_models/best.onnx",
		image_path="test_img/stop.jpg",
        imgsz=640,
		save_dir="test_img/results",
    )
	

__all__ = ["StopSignDetectionModel"]
    