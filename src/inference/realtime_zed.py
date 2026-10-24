import os
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from ultralytics import YOLO
from pathlib import Path

import pyzed.sl as sl  # ZED SDK Python

from src.models.efficientnet_cbam import RGBDEfficientNet

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMOTION_MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "finetuned_fold7_best.pth"
YOLO_FACE_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_face.pt"

SAVE_DIR = PROJECT_ROOT / "outputs"
SAVE_ANNOTATED = False  # en temps reel: souvent False
CSV_NAME = "results_realtime.csv"

# Lissage temporel
BUFFER_LEN = 5

# Conversion depth (metres) -> grayscale u8
# (tout ce qui est au-dela devient noir, ajuste selon ta scene)
MAX_DEPTH_M = 3.0
# ==================================================


def add_margin(x1, y1, x2, y2, margin=0.2):
    bw, bh = (x2 - x1), (y2 - y1)
    mx, my = int(bw * margin), int(bh * margin)
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def make_rgbd_tensor(rgb_crop_bgr, depth_crop_u8):
    # RGB -> 0..1
    rgb = cv2.cvtColor(rgb_crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0

    # Depth u8 -> 0..1
    d = cv2.resize(depth_crop_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    d = d.astype(np.float32) / 255.0
    d = d[..., None]

    x = np.concatenate([rgb, d], axis=2)  # (224,224,4)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return x


def pick_biggest_face(boxes):
    areas = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        areas.append((x2 - x1) * (y2 - y1))
    return int(np.argmax(areas))


def depth_to_u8(depth_m, max_depth_m=MAX_DEPTH_M):
    """
    depth_m: array float32 en metres (ZED MEASURE.DEPTH)
    -> grayscale u8 (proche = clair, loin = sombre)
    """
    d = depth_m.copy()
    # invalid -> max
    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = max_depth_m

    d = np.clip(d, 0, max_depth_m)
    # inverse: proche clair
    d_norm = 1.0 - (d / max_depth_m)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    return d_u8


def main():
    # 1) Charger YOLO face
    yolo = YOLO(YOLO_FACE_WEIGHTS)
    print("✅ YOLO face chargé:", YOLO_FACE_WEIGHTS)

    # 2) Charger modèle émotions
    model = RGBDEfficientNet(num_classes=7).to(DEVICE)
    state = torch.load(EMOTION_MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("✅ Modèle émotions chargé:", EMOTION_MODEL_WEIGHTS)

    # 3) Sorties
    os.makedirs(SAVE_DIR, exist_ok=True)
    ann_dir = os.path.join(SAVE_DIR, "annotated_realtime")
    if SAVE_ANNOTATED:
        os.makedirs(ann_dir, exist_ok=True)

    csv_path = os.path.join(SAVE_DIR, CSV_NAME)
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow([
        "frame_idx", "pred_label", "confidence",
        "p0","p1","p2","p3","p4","p5","p6",
        "bbox", "buffer_len"
    ])

    # 4) Init ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        csv_f.close()
        raise RuntimeError(f"❌ ZED open failed: {repr(status)}")

    runtime = sl.RuntimeParameters()
    # Tu peux filtrer/ameliorer la depth si besoin:
    # runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    image_left = sl.Mat()
    depth_measure = sl.Mat()

    # buffer moyenne
    prob_buffer = deque(maxlen=BUFFER_LEN)

    frame_idx = 0
    t0 = cv2.getTickCount()

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)                 # BGRA
            zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)        # float32 metres

            left_bgra = image_left.get_data()
            frame = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)

            depth_m = depth_measure.get_data()  # HxWx1 (ou HxW) float32
            if depth_m.ndim == 3:
                depth_m = depth_m[..., 0]
            depth_u8 = depth_to_u8(depth_m, MAX_DEPTH_M)

            h, w = frame.shape[:2]

            # YOLO sur RGB
            results = yolo(frame, verbose=False)
            boxes = results[0].boxes

            label_to_show = "NoFace"
            bbox_str = ""

            if boxes is not None and len(boxes) > 0:
                idx = pick_biggest_face(boxes)
                x1, y1, x2, y2 = boxes[idx].xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = add_margin(x1, y1, x2, y2, margin=0.2)
                bb = clip_bbox(x1, y1, x2, y2, w, h)

                if bb is not None:
                    x1, y1, x2, y2 = bb
                    bbox_str = f"{x1},{y1},{x2},{y2}"

                    rgb_crop = frame[y1:y2, x1:x2]
                    d_crop = depth_u8[y1:y2, x1:x2]

                    x_in = make_rgbd_tensor(rgb_crop, d_crop)

                    with torch.no_grad():
                        logits = model(x_in)
                        p = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

                    prob_buffer.append(p)
                    p_avg = np.mean(prob_buffer, axis=0)

                    pred = int(np.argmax(p_avg))
                    conf = float(np.max(p_avg))
                    label_to_show = f"{EMOTIONS[pred]} ({conf:.2f})"

                    writer.writerow([
                        frame_idx, EMOTIONS[pred], f"{conf:.4f}",
                        f"{p_avg[0]:.6f}", f"{p_avg[1]:.6f}", f"{p_avg[2]:.6f}",
                        f"{p_avg[3]:.6f}", f"{p_avg[4]:.6f}", f"{p_avg[5]:.6f}",
                        f"{p_avg[6]:.6f}", bbox_str, len(prob_buffer)
                    ])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    writer.writerow([frame_idx, "BadBBox", "0.0000", "", "", "", "", "", "", "", bbox_str, len(prob_buffer)])
            else:
                writer.writerow([frame_idx, "NoFace", "0.0000", "", "", "", "", "", "", "", bbox_str, len(prob_buffer)])

            # FPS
            t1 = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(1.0, (t1 - t0))
            t0 = t1

            cv2.putText(frame, f"Frame {frame_idx} | {label_to_show} | FPS {fps:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # (optionnel) afficher aussi la depth
            # depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            # cv2.imshow("Depth (u8)", depth_vis)

            if SAVE_ANNOTATED:
                cv2.imwrite(os.path.join(ann_dir, f"frame_{frame_idx:06d}.png"), frame)

            cv2.imshow("ZED + YOLOv8 Face + Emotion (avg)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC ou q
                break
            if key == ord('c'):  # clear buffer
                prob_buffer.clear()

            frame_idx += 1

    finally:
        csv_f.close()
        cv2.destroyAllWindows()
        zed.close()
        print("✅ CSV sauvegardé:", csv_path)
        if SAVE_ANNOTATED:
            print("✅ Images annotées:", ann_dir)


if __name__ == "__main__":
    main()