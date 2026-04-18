from ultralytics import YOLO

model = YOLO("runs/detect/yolo8nretrain/weights/best.pt")

model.export(
    format="onnx",
    opset=12,
    imgsz=640,
    half=True
)