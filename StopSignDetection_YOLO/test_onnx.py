from ultralytics import YOLO
from pathlib import Path

onnx_model_path = Path("StopSignDetection_YOLO/runs/detect/finetune_stage1/weights/best.onnx")
img_path = Path("StopSignDetection_YOLO/test_img/stop.jpg")

m = YOLO(str(onnx_model_path))

# run prediction; adjust imgsz, conf, iou as needed
results = m.predict(source=str(img_path), imgsz=640, conf=0.25, iou=0.45, device='cpu', save=True)

# # results is a list-like object; show/save/print
# for r in results:
#     print(r.boxes.xyxy)   # bounding boxes (xyxy)
#     print(r.boxes.conf)   # confidences
#     print(r.boxes.cls)    # class ids
# # Save prediction image(s) with boxes (in working dir)
# results.save()  # writes annotated images to ./runs/detect/predict by default