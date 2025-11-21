import onnxruntime as ort
import cv2
import numpy as np

# ---- Config ----
MODEL_PATH = "onnx_model_old/culane_res34.onnx"
TEST_IMAGE = "test.png"  # adjust path if needed

IMG_W, IMG_H = 1600, 320

# Normalization parameters (assumed ImageNet-like)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Anchor configuration (approximate)
NUM_ROW_ANCHORS = 72
NUM_COL_ANCHORS = 81
row_anchor_positions = np.linspace(0, IMG_H - 1, NUM_ROW_ANCHORS)
col_anchor_positions = np.linspace(0, IMG_W - 1, NUM_COL_ANCHORS)

EXIST_THRESH = 0.5

# ---- ONNX Runtime setup ----
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print("Providers:", session.get_providers())
print("Input name:", input_name)
print("Output names:", output_names)


# ---- Preprocess ----
def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # B = 1
    return img


# ---- Softmax / Sigmoid ----
def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---- Decode function ----
def decode_lanes(loc_row, loc_col, exist_row, exist_col):
    """
    Decode lane positions from model outputs.
    loc_row: [1, bins_row, num_row_anchors, num_lanes]
    loc_col: [1, bins_col, num_col_anchors, num_lanes]
    exist_row: [1, 2, num_row_anchors, num_lanes]
    exist_col: [1, 2, num_col_anchors, num_lanes]
    """
    loc_row = loc_row[0]
    loc_col = loc_col[0]
    exist_row = exist_row[0]
    exist_col = exist_col[0]

    num_bins_row, num_row_anchors, num_lanes = loc_row.shape
    num_bins_col, num_col_anchors, _ = loc_col.shape

    # probabilities
    prob_row = softmax(loc_row, axis=0)
    prob_col = softmax(loc_col, axis=0)
    exist_row_prob = sigmoid(exist_row[1, :, :])  # assume index 1 = “exists”
    exist_col_prob = sigmoid(exist_col[1, :, :])

    lanes = []
    for lane_idx in range(num_lanes):
        pts = []
        # row anchors
        for r in range(num_row_anchors):
            if exist_row_prob[r, lane_idx] > EXIST_THRESH:
                # expectation for row
                bins = prob_row[:, r, lane_idx]
                expected_bin = np.sum(bins * np.arange(num_bins_row))
                norm_x = expected_bin / (num_bins_row - 1)
                x = norm_x * (IMG_W - 1)
                y = row_anchor_positions[r]
                pts.append((x, y))
        # optionally also use column anchors
        # but here we focus on row-based decoding

        # fit a polynomial if enough points
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            poly = np.polyfit(ys, xs, deg=2)
            ys_smooth = np.linspace(min(ys), max(ys), 100)
            xs_smooth = np.polyval(poly, ys_smooth)
            lane_curve = [(int(x), int(y)) for x, y in zip(xs_smooth, ys_smooth)]
        else:
            lane_curve = [(int(x), int(y)) for x, y in pts]

        if lane_curve:
            lanes.append(lane_curve)
    return lanes


# ---- Draw lanes ----
def draw_lanes(img, lanes):
    for lane in lanes:
        for i in range(len(lane) - 1):
            pt1 = (int(lane[i][0]), int(lane[i][1]))
            pt2 = (int(lane[i + 1][0]), int(lane[i + 1][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), thickness=2)
    return img


# ---- Main ----
def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    input_tensor = preprocess(img)
    outputs = session.run(output_names, {input_name: input_tensor})

    loc_row, loc_col, exist_row, exist_col = outputs
    lanes = decode_lanes(loc_row, loc_col, exist_row, exist_col)
    print("Decoded lanes (pixel pts):", lanes)

    vis = img.copy()
    vis = draw_lanes(vis, lanes)
    cv2.imwrite("warp_image/lanes_" + TEST_IMAGE, vis)
    print("Saved decoded image to decoded_lanes.png")
    return vis, lanes


if __name__ == "__main__":
    result = main(TEST_IMAGE)
