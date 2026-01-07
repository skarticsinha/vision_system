import cv2
import csv
import numpy as np
from pathlib import Path

from vision_system.utils.embedder import MobileNetEmbedder
from vision_system.utils.similarity import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "embeddings.csv"

SIM_THRESHOLD = 0.75

def load_embeddings():
    database = []
    with open(DATASET_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            obj = row[0]
            defect = row[1]
            emb = np.array(row[2:], dtype=np.float32)
            database.append((obj, defect, emb))
    return database

def main():
    database = load_embeddings()
    print(f"Loaded {len(database)} embeddings")

    embedder = MobileNetEmbedder()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emb = embedder.extract(frame)

        best_score = -1
        best_match = None

        for obj, defect, ref_emb in database:
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_match = (obj, defect)

        if best_score >= SIM_THRESHOLD:
            label = f"{best_match[0]} | Defect: {best_match[1]} | {best_score:.2f}"
            color = (0, 255, 0)
        else:
            label = f"Unknown | {best_score:.2f}"
            color = (0, 0, 255)

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
