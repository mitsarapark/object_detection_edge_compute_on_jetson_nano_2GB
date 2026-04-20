from ultralytics import YOLO
import os
if __name__ == "__main__":
    YOLO("~/object_detection_edge_compute_on_jetson_nano_2GB/models/yolov8n.pt").train(
        data="/data.yaml",#data coco dataset after run splitratio.py
        epochs=30,
        device=0,
        batch=16,
        imgsz=640,
        workers=8,
        name="yolo8nretrain_30_time"
    )