"""
YOLOv8 TensorRT Validation Script
- Precision, Recall, mAP50, mAP50-95
- Python 3.6 + TensorRT (JetPack 4.6.6) on Jetson Nano 2GB
- VOC dataset converted to YOLO format
- 5 classes: person, bicycle, car, motorcycle, bus
- ใช้ Letterbox resize (รักษา aspect ratio)

Usage:
  python3.6 validate_yolov8_trt.py \
    --engine model.engine \
    --data /path/to/images \
    --labels /path/to/labels \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45
"""

import argparse
import os
import time
import glob
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus"]
NUM_CLASSES = len(CLASS_NAMES)  # = 5


# ──────────────────────────────────────────
# TensorRT Engine Loader
# ──────────────────────────────────────────
def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


class TRTInferencer:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def infer(self, img_array):
        np.copyto(self.inputs[0]["host"], img_array.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        return [out["host"].reshape(out["shape"]) for out in self.outputs]


def letterbox_preprocess(image_path, imgsz=640):
    img = cv2.imread(image_path)
    if img is None:
        print("  [WARN] Cannot read: {}".format(image_path))
        return None, None, None, None, None, None

    orig_h, orig_w = img.shape[:2]

    scale = min(imgsz / orig_w, imgsz / orig_h)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)

    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR→RGB, HWC→CHW, normalize, add batch dim
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_batch = np.expand_dims(blob, axis=0)

    return np.ascontiguousarray(img_batch), orig_w, orig_h, scale, pad_x, pad_y


def postprocess(output, orig_w, orig_h, scale, pad_x, pad_y,
                conf_thres=0.25, iou_thres=0.45):
    pred = output[0]   # [9, 8400]
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T      # [8400, 9]

    boxes = pred[:, :4]                # cx, cy, w, h (ใน letterbox space)
    scores = pred[:, 4:4+NUM_CLASSES]  # class scores

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = np.clip(((cx - bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y1 = np.clip(((cy - bh / 2) - pad_y) / scale, 0, orig_h).astype(int)
    x2 = np.clip(((cx + bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y2 = np.clip(((cy + bh / 2) - pad_y) / scale, 0, orig_h).astype(int)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = confidences.tolist()
    class_ids = class_ids.tolist()

    # NMS
    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_xyxy],confidences, conf_thres, iou_thres
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "box": [float(v) for v in boxes_xyxy[i]],"confidence": confidences[i],"class_id": int(class_ids[i])
            })
    return detections


def load_labels(label_path, orig_w, orig_h):
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= NUM_CLASSES:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            gts.append({"box": [x1, y1, x2, y2], "class_id": cls_id})
    return gts


# ──────────────────────────────────────────
# IoU
# ──────────────────────────────────────────
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────
# Compute AP
# ──────────────────────────────────────────
def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def compute_map(all_detections, all_ground_truths, iou_threshold=0.5):
    aps = []
    ap_per_class = {}

    for cls_id in range(NUM_CLASSES):
        tp_list, conf_list = [], []
        num_gt = 0

        for dets, gts in zip(all_detections, all_ground_truths):
            gt_cls = [g for g in gts if g["class_id"] == cls_id]
            det_cls = sorted(
                [d for d in dets if d["class_id"] == cls_id],
                key=lambda x: -x["confidence"]
            )
            num_gt += len(gt_cls)
            matched = [False] * len(gt_cls)

            for det in det_cls:
                conf_list.append(det["confidence"])
                best_iou, best_j = 0, -1
                for j, gt in enumerate(gt_cls):
                    iou = compute_iou(det["box"], gt["box"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                    tp_list.append(1)
                    matched[best_j] = True
                else:
                    tp_list.append(0)

        if num_gt == 0:
            ap_per_class[CLASS_NAMES[cls_id]] = 0.0
            continue

        tp_arr = np.array(tp_list)[np.argsort(-np.array(conf_list))]
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(1 - tp_arr)
        recalls = tp_cum / (num_gt + 1e-16)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        ap_per_class[CLASS_NAMES[cls_id]] = ap

    return np.mean(aps) if aps else 0.0, ap_per_class


def validate(args):
    print("\n" + "="*60)
    print("  YOLOv8 TensorRT Validation (5-class VOC + Letterbox)")
    print("="*60)
    print("  Engine  :", args.engine)
    print("  Images  :", args.data)
    print("  Labels  :", args.labels)
    print("  Classes :", CLASS_NAMES)
    print("  ImgSize :", args.imgsz)
    print("  Conf    :", args.conf)
    print("  IoU     :", args.iou)
    print("="*60 + "\n")

    print("[1/2] Loading TensorRT engine")
    inferencer = TRTInferencer(args.engine)
    print("      Engine loaded OK\n")

    image_paths = sorted(
        glob.glob(os.path.join(args.data, "*.jpg")) +
        glob.glob(os.path.join(args.data, "*.jpeg")) +
        glob.glob(os.path.join(args.data, "*.png"))
    )
    print("Found {} images\n".format(len(image_paths)))

    all_detections, all_ground_truths, inference_times = [], [], []

    print("[2/2] Running inference...")
    for idx, img_path in enumerate(image_paths):

        # ── Letterbox preprocess ──
        result = letterbox_preprocess(img_path, args.imgsz)
        img_batch, orig_w, orig_h, scale, pad_x, pad_y = result
        if img_batch is None:
            continue

        t0 = time.time()
        outputs = inferencer.infer(img_batch)
        inference_times.append((time.time() - t0) * 1000)

        dets = postprocess(outputs, orig_w, orig_h,scale, pad_x, pad_y,args.conf, args.iou)#post process output

        label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        gts = load_labels(args.labels+"/"+label_file, orig_w, orig_h)#process label

        all_detections.append(dets)
        all_ground_truths.append(gts)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print("  [{}/{}] {:.1f} ms/img".format(idx+1, len(image_paths), np.mean(inference_times[-10:])))

    # ── Metrics ──
    print("\n" + "="*60)
    print("  Computing Metrics...")

    map50, ap50_per_class = compute_map(all_detections, all_ground_truths, 0.50)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)

    map50_95 = np.mean([compute_map(all_detections, all_ground_truths, t)[0]for t in iou_thresholds
                        ])

    total_tp = total_fp = total_fn = 0
    for dets, gts in zip(all_detections, all_ground_truths):
        matched_gt = set()
        for det in dets:
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gts):
                if gt["class_id"] != det["class_id"]:
                    continue
                iou = compute_iou(det["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j not in matched_gt:
                total_tp += 1
                matched_gt.add(best_j)
            else:
                total_fp += 1
        total_fn += len(gts) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-16)#1e-16 because if total_tp+total_fp=0 it will error
    recall = total_tp / (total_tp + total_fn + 1e-16)#1e-16 = epsilon
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    avg_inf = np.mean(inference_times)

    print("\n  {:<20} {:>10}".format("Metric", "Value"))
    print("  " + "-"*32)
    print("  {:<20} {:>10.4f}".format("Precision",    precision))
    print("  {:<20} {:>10.4f}".format("Recall",       recall))
    print("  {:<20} {:>10.4f}".format("F1-Score",     f1))
    print("  {:<20} {:>10.4f}".format("mAP@0.5",      map50))
    print("  {:<20} {:>10.4f}".format("mAP@0.5:0.95", map50_95))
    print("  " + "-"*32)
    print("\n  AP per class (IoU=0.5):")
    for name, ap in ap50_per_class.items():
        print("    {:<15} {:.4f}".format(name, ap))
    print("  " + "-"*32)
    print("  {:<20} {:>9.1f}ms".format("Avg Inference", avg_inf))
    print("  {:<20} {:>9.1f}".format("FPS",             1000.0/avg_inf))
    print("  {:<20} {:>10}".format("Total Images",  len(image_paths)))
    print("="*60 + "\n")

    with open("validation_results.txt", "w") as f:
        f.write("YOLOv8 TensorRT Validation Results\n")
        f.write("Engine: {}\n".format(args.engine))
        f.write("Classes: {}\n".format(CLASS_NAMES))
        f.write("Images: {}\n\n".format(len(image_paths)))
        f.write("Precision:    {:.4f}\n".format(precision))
        f.write("Recall:       {:.4f}\n".format(recall))
        f.write("F1-Score:     {:.4f}\n".format(f1))
        f.write("mAP@0.5:      {:.4f}\n".format(map50))
        f.write("mAP@0.5:0.95: {:.4f}\n".format(map50_95))
        f.write("\nAP per class:\n")
        for name, ap in ap50_per_class.items():
            f.write("  {}: {:.4f}\n".format(name, ap))
        f.write("\nAvg Inference: {:.1f} ms\n".format(avg_inf))
        f.write("FPS: {:.1f}\n".format(1000.0/avg_inf))
    print("  Results saved to: validation_results.txt\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",  required=True)
    parser.add_argument("--data",    required=True,  help="Path to images dir")
    parser.add_argument("--labels",  required=True,  help="Path to labels dir")
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.5)
    args = parser.parse_args()
    validate(args)

#LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 validation.py --engine model_retrain_fp16.engine --data voc_yolo/images/val --labels voc_yolo/labels/val