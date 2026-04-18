from ultralytics import YOLO
model = YOLO("runs/detect/yolo8nretrain/weights/best.pt")

metrics = model.val(
data="voc_yolo/data.yaml",
imgsz=640,
batch=1,
device=0
)

print(metrics)
