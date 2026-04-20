from ultralytics import YOLO

model = YOLO("~/object_detection_edge_compute_on_jetson_nano_2GB/models/yolo8nretrain/weights/best.pt")

model.export(
    format="onnx",
    opset=12,
    imgsz=640,
    half=True
)