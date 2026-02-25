import cv2
import argparse

from . import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--config_path', 
            type=str,
            default='configs/tusimple_res34.py',
            help="Path to the configuration file for the model"
    )
    parser.add_argument(
            '--onnx_path', 
            type=str,
            default='weights/tusimple_res34.onnx', 
            help="Path to the ONNX model"
    )
    parser.add_argument(
            '--video_path', 
            type=str,
            default='test.png', 
            help="Path to the video to predict"
    )
    parser.add_argument(
            '--ori_size', 
            nargs=2, 
            type=int,
            default=(1280, 720),
            help="Dimensions of the original image"
    )
    return parser.parse_args()


args = get_args()
print(f"Reading {args.video_path}...")
cap = cv2.VideoCapture(args.video_path)
#model = UFLDv2_ONNX(args.onnx_path, args.config_path, args.ori_size)
model = UFLDv2_ONNX(args.ori_size)

while True:
    success, _img = cap.read()
    if not success:
        break
    img = _img

    coords, lane_exists = model(img)

    print()
    for i in range(4):
        if len(coords[i]) > 0:
            print(f"[{i}] len: {len(coords[i])} | delta: {np.diff(np.array(coords[i])[:, 0]).mean()}")

    # ---- DRAW RESULTS ----
    for exists, lane, color in zip(lane_exists, coords, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]):
        if not exists:
            continue

        for (x, y) in lane:
            cv2.circle(img, (x, y), 2, color, -1)

    cv2.imshow("result", img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

