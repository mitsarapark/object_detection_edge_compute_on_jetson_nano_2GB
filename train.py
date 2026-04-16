from ultralytics import YOLO
import os
if __name__ == "__main__":
    YOLO("yolov8n.pt").train(
        data="D:/NSTDA/final_merge/data.yaml",
        epochs=30,
        device=0,
        batch=16,
        imgsz=640,
        workers=8,
        name="yolo8nretrain_30_time"
    )