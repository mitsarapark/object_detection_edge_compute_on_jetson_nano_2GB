from ultralytics import YOLO
model = YOLO("~/object_detection_edge_compute_on_jetson_nano_2GB/models/yolo8nretrain/weights/best.pt")

metrics = model.val(
data="voc_yolo/data.yaml",
imgsz=640,
batch=1,
device=0
)

print(metrics)
