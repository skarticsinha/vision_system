import cv2
import csv
import numpy as np
from pathlib import Path
from collections import deque, Counter

from ultralytics import YOLO
from vision_system.utils.embedder import MobileNetEmbedder
from vision_system.utils.similarity import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "embeddings.csv"

YOLO_CONF_THRESHOLD = 0.5
SIM_THRESHOLD = 0.8
VOTING_WINDOW = 10

# --------------------------------------------------
def load_embeddings():
    db = []
    with open(DATASET_PATH, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            db.append((r[0], r[1], np.array(r[2:], dtype=np.float32)))
    return db

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

    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    idx = areas.argmax()

    mask = masks[idx]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask > 0.5).astype("uint8")

    ys, xs = np.where(mask == 1)
    if len(xs)==0 or len(ys)==0:
        return None, None

    x1,x2 = xs.min(), xs.max()
    y1,y2 = ys.min(), ys.max()

    masked = cv2.bitwise_and(frame, frame, mask=mask*255)
    roi = masked[y1:y2, x1:x2]

    return roi, (x1,y1,x2,y2)

# --------------------------------------------------
def main():
    db = load_embeddings()
    embedder = MobileNetEmbedder()
    yolo = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    votes = deque(maxlen=VOTING_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, bbox = get_object_roi(frame, yolo)
        votes.append("NoObject")

        if roi is not None:
            emb = embedder.extract(roi)
            best = max(db, key=lambda x: cosine_similarity(emb, x[2]))
            score = cosine_similarity(emb, best[2])

            if score >= SIM_THRESHOLD:
                votes[-1] = f"{best[0]}|{best[1]}"

            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)

        label, count = Counter(votes).most_common(1)[0]

        if label=="NoObject":
            text="No object"
            color=(0,0,255)
        else:
            o,d = label.split("|")
            text=f"{o} | Defect {d}"
            color=(0,255,0)

        cv2.putText(frame,text,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        cv2.imshow("Detection (SEGMENTATION)", frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
