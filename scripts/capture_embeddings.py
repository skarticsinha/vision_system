import cv2
import csv
import time
from pathlib import Path

from vision_system.utils.embedder import MobileNetEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "embeddings.csv"

NUM_FRAMES = 80
CAPTURE_DELAY = 0.05

def main():
    object_name = input("Enter object name / ID: ").strip()
    defect_level = input("Defect level (0=no, 1=minor, 2=major): ").strip()

    if defect_level not in {"0", "1", "2"}:
        print("Invalid defect level")
        return

    input("Press ENTER and rotate the object in front of the camera...")

    embedder = MobileNetEmbedder()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    with open(DATASET_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        for i in range(NUM_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break

            embedding = embedder.extract(frame)
            row = [object_name, defect_level] + embedding.tolist()
            writer.writerow(row)

            cv2.imshow("Capture", frame)
            print(f"Captured {i+1}/{NUM_FRAMES}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(CAPTURE_DELAY)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Capture complete. Embeddings saved.")

if __name__ == "__main__":
    main()
