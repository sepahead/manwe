from ultralytics import YOLO
import sys

print("Loading YOLOv8s PyTorch model...")
model = YOLO("yolov8s.pt")

print("Exporting to CoreML (FP32)...")
# nms=True adds the Non-Maximum Suppression layers directly into the CoreML model
# half=False ensures FP32 precision
model.export(format="coreml", half=False, nms=True)

print("Export complete: yolov8s.mlpackage")
