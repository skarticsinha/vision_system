import cv2
import csv
import time
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from vision_system.utils.embedder import MobileNetEmbedder

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "embeddings.csv"

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
NUM_FRAMES = 80
CAPTURE_DELAY = 0.05
YOLO_CONF_THRESHOLD = 0.5

# --------------------------------------------------
# SEGMENTATION ROI
# --------------------------------------------------
def get_object_roi(frame, yolo):
    results = yolo(frame, verbose=False)

    if not results or results[0].masks is None:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    masks = results[0].masks.data.cpu().numpy()

    valid = scores >= YOLO_CONF_THRESHOLD
    if not valid.any():
        return None, None

    boxes = boxes[valid]
    masks = masks[valid]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = areas.argmax()

    mask = masks[idx]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask > 0.5).astype("uint8")

    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    masked = cv2.bitwise_and(frame, frame, mask=mask * 255)
    roi = masked[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    return roi, (x1, y1, x2, y2)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    obj_name = input("Enter object name: ").strip()
    defect = input("Defect level (0/1/2): ").strip()

    if defect not in {"0", "1", "2"}:
        print("Invalid defect level")
        return

    input("Press ENTER and rotate object slowly...")

    embedder = MobileNetEmbedder()
    yolo = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    captured = 0

    with open(DATASET_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while captured < NUM_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            roi_data = get_object_roi(frame, yolo)

            if roi_data is None:
                cv2.putText(frame, "No object", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Capture", frame)
                cv2.waitKey(1)
                continue

            roi, (x1,y1,x2,y2) = roi_data
            emb = embedder.extract(roi)

            writer.writerow([obj_name, defect] + emb.tolist())
            captured += 1

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{captured}/{NUM_FRAMES}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Capture", frame)
            cv2.waitKey(1)
            time.sleep(CAPTURE_DELAY)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Capture complete (SEGMENTATION-BASED)")

if __name__ == "__main__":
    main()
