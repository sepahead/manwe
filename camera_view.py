#!/opt/anaconda3/envs/py3-14/bin/python
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configuration
RTSP_URLS = [
    # "rtsp://admin:sauronsauron1@192.168.10.172:554/h264",
    # "rtsp://admin:sauronsauron1@192.168.10.145:554/h264"
    "rtsp://root:root@192.168.10.100/axis-media/media.amp"
    # "rtsp://root:root@192.168.10.100/axis-media/media.amp"
]
MODEL_NAME = "yolov8n.pt"


class StreamThread(threading.Thread):
    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        # Load model per thread to ensure thread safety and isolation
        # M4 Max is fast enough to run YOLOv8n on every frame without skipping
        self.model = YOLO(MODEL_NAME)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Stream {index} using device: {self.device}")

    def run(self):
        print(f"Connecting to {self.url}...")
        cap = cv2.VideoCapture(self.url)

        # Optimize buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {self.index} failed to read frame. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.url)
                continue

            # Inference
            # verbose=False prevents spamming stdout
            results = self.model(frame, device=self.device, verbose=False)

            # Plot results on the frame
            annotated_frame = results[0].plot()

            with self.lock:
                self.frame = annotated_frame

        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.join()


def main():
    threads = []
    for i, url in enumerate(RTSP_URLS):
        t = StreamThread(url, i)
        t.start()
        threads.append(t)

    print("Streams started. Press 'q' to quit.")

    try:
        while True:
            frames = []
            for t in threads:
                f = t.get_frame()
                if f is None:
                    # Placeholder if no frame yet
                    f = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        f,
                        "Connecting...",
                        (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # Resize to standard size for display
                    f = cv2.resize(f, (640, 360))
                frames.append(f)

            # Concatenate side by side
            if frames:
                combined = np.hstack(frames)
                cv2.imshow("Multi-Stream YOLO (PyTorch)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Small sleep to prevent busy loop if waitKey is fast
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping threads...")
        for t in threads:
            t.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
