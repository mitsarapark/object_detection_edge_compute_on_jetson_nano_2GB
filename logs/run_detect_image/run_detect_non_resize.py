"""
YOLOv8n TensorRT Image Detection
Output shape: [1, 9, 8400]  →  9 = 4 (cx,cy,w,h) + 5 classes
Usage:
    python detect_image.py --engine model.engine --image photo.jpg
    python detect_image.py --engine model.engine --image photo.jpg --conf 0.5 --labels labels.txt
"""

import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import time


# ─── TensorRT Logger ─────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ─── Engine Loader ───────────────────────────────────────────────────────────

def load_engine(engine_path: str):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# ─── Buffer Allocation ───────────────────────────────────────────────────────

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream


# ─── Inference ───────────────────────────────────────────────────────────────

def infer(context, inputs, outputs, bindings, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    return [out["host"] for out in outputs]


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess(image_path: str, input_h: int, input_w: int):
    """
    Letterbox resize → RGB → CHW → float32 → [0,1]
    Returns: blob (C,H,W), original image, scale, pad_x, pad_y
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(input_w / orig_w, input_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), img_bgr, scale, pad_x, pad_y


# ─── Postprocessing ──────────────────────────────────────────────────────────

def postprocess_yolov8(raw, orig_w, orig_h, scale, pad_x, pad_y,
                       num_classes=5, conf_thresh=0.5, iou_thresh=0.45):
    """
    Parse YOLOv8 output shape [1, 9, 8400]
      - dim 1 row 0-3 : cx, cy, w, h  (in letterbox pixel space)
      - dim 1 row 4-8 : class scores  (no objectness score in YOLOv8)
    Returns list of (x1, y1, x2, y2, conf, class_id)
    """
    # raw is flat → reshape to (9, 8400) then transpose to (8400, 9)
    preds = raw.reshape(4 + num_classes, 8400).T   # (8400, 9)

    cx  = preds[:, 0]
    cy  = preds[:, 1]
    bw  = preds[:, 2]
    bh  = preds[:, 3]
    cls_scores = preds[:, 4:]                       # (8400, 5)

    class_ids = np.argmax(cls_scores, axis=1)
    confs     = cls_scores[np.arange(len(cls_scores)), class_ids]

    # Filter by confidence
    mask = confs >= conf_thresh
    if not mask.any():
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confs     = confs[mask]
    class_ids = class_ids[mask]

    # Convert letterbox coords → original image coords
    x1 = np.clip(((cx - bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y1 = np.clip(((cy - bh / 2) - pad_y) / scale, 0, orig_h).astype(int)
    x2 = np.clip(((cx + bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y2 = np.clip(((cy + bh / 2) - pad_y) / scale, 0, orig_h).astype(int)

    # NMS
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), conf_thresh, iou_thresh)

    detections = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            detections.append((x1[i], y1[i], x2[i], y2[i],
                               float(confs[i]), int(class_ids[i])))
    return detections


# ─── Drawing ─────────────────────────────────────────────────────────────────

PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134), ( 26, 147,  52),
    (  0, 194, 255), ( 52,  69, 147), (100, 115, 255), (132,  56, 255),
]

def draw_detections(image, detections, labels=None):
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = labels[cls_id] if labels and cls_id < len(labels) else f"class {cls_id}"
        text  = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n TensorRT — detect on image")
    parser.add_argument("--engine",      required=True,         help="Path to .engine file")
    parser.add_argument("--image",       required=True,         help="Path to input image")
    parser.add_argument("--output",      default="result.jpg",  help="Output image path")
    parser.add_argument("--conf",        type=float, default=0.5)
    parser.add_argument("--iou",         type=float, default=0.45)
    parser.add_argument("--num-classes", type=int,   default=5, help="Number of classes")
    parser.add_argument("--labels",      default=None,          help="labels.txt — one class per line")
    parser.add_argument("--input-size",  type=int,   default=640)
    parser.add_argument("--no-show",     action="store_true",   help="headless — don't open window")
    args = parser.parse_args()

    # Load labels
    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = [l.strip() for l in f.readlines()]
        print(f"[INFO] Classes: {labels}")

    # Load engine
    print(f"[INFO] Loading engine: {args.engine}")
    engine  = load_engine(args.engine)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Debug binding shapes
    for i, binding in enumerate(engine):
        shape = engine.get_binding_shape(binding)
        kind  = "INPUT" if engine.binding_is_input(binding) else "OUTPUT"
        print(f"[INFO] Binding [{i}] '{binding}'  shape={list(shape)}  {kind}")

    input_size = args.input_size

    # Preprocess
    print(f"[INFO] Image: {args.image}")
    blob, orig_img, scale, pad_x, pad_y = preprocess(
        args.image, input_size, input_size)
    np.copyto(inputs[0]["host"], blob.flatten())

    # Infer
    t0 = time.perf_counter()
    raw_outputs = infer(context, inputs, outputs, bindings, stream)
    t1 = time.perf_counter()
    print(f"[INFO] Inference: {(t1 - t0) * 1000:.1f} ms")

    # Postprocess
    orig_h, orig_w = orig_img.shape[:2]
    detections = postprocess_yolov8(
        raw_outputs[0],
        orig_w, orig_h, scale, pad_x, pad_y,
        num_classes=args.num_classes,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    # Print results
    print(f"[INFO] Detections found: {len(detections)}")
    for x1, y1, x2, y2, conf, cls_id in detections:
        name = labels[cls_id] if labels and cls_id < len(labels) else f"class_{cls_id}"
        print(f"       {name:20s}  conf={conf:.3f}  box=[{x1},{y1},{x2},{y2}]")

    # Draw & save
    result = draw_detections(orig_img.copy(), detections, labels)
    cv2.imwrite(args.output, result)
    print(f"[INFO] Saved → {args.output}")

    if not args.no_show:
        cv2.imshow("YOLOv8 TensorRT", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# basic
#python detect_image.py --engine best32.engine --image crosswalk.jpg

# พร้อม labels + headless (ไม่มี display)
#python detect_image.py --engine model.engine --image photo.jpg --labels labels.txt --conf 0.5 --no-show