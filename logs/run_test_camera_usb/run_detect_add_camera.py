"""
YOLOv8n TensorRT — Video / Camera Detection (Letterbox)
Output shape: [1, 9, 8400]  →  9 = 4 (cx,cy,w,h) + 5 classes

Usage:
    # วิดีโอไฟล์
    python detect_video.py --engine model.engine --source input.mp4

    # กล้อง (CSI camera บน Jetson Nano)
    python detect_video.py --engine model.engine --source csi

    # กล้อง USB (device index 0)
    python detect_video.py --engine model.engine --source 0

    # headless (ไม่มี monitor)
    python detect_video.py --engine model.engine --source 0 --no-show
"""

import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time


# ─── TensorRT Logger ─────────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ─── Engine Loader ───────────────────────────────────────────────────────────
def load_engine(engine_path: str):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# ─── Buffer Allocation ───────────────────────────────────────────────────────
def allocate_buffers(engine):
    """จอง pinned CPU memory + GPU memory สำหรับทุก binding"""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size  = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    return inputs, outputs, bindings, stream


# ─── Inference ───────────────────────────────────────────────────────────────

def infer(context, inputs, outputs, bindings, stream):
    """CPU→GPU → run model → GPU→CPU"""
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()
    return [out["host"] for out in outputs]


# ─── Camera Open Helper ──────────────────────────────────────────────────────
def open_source(source: str, width: int, height: int, fps: int):

    if source == "csi":
        # GStreamer pipeline สำหรับ CSI camera บน Jetson Nano
        # nvarguscamerasrc = ISP hardware ของ Jetson, ให้คุณภาพดีกว่า v4l2
        pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cap, True

    # ลองแปลง source เป็น int เพื่อดูว่าเป็น device index หรือเปล่า
    try:
        idx = int(source)
        # USB camera — ใช้ V4L2 backend ตรงๆ เร็วกว่า auto-detect
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        # ตั้ง resolution และ FPS ที่ต้องการ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        # ลด buffer เหลือ 1 frame เพื่อลด latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, True
    except ValueError:
        cap = cv2.VideoCapture(source)
        return cap, False


# ─── Letterbox ───────────────────────────────────────────────────────────────

def letterbox_frame(frame_bgr, input_size: int):

    orig_h, orig_w = frame_bgr.shape[:2]

    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)

    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), scale, pad_x, pad_y


# ─── Postprocessing ──────────────────────────────────────────────────────────

def postprocess_yolov8(raw, orig_w, orig_h, scale, pad_x, pad_y,
                       num_classes=5, conf_thresh=0.5, iou_thresh=0.45):
    """แปลง raw output [1, 9, 8400] → list of (x1, y1, x2, y2, conf, class_id)"""
    preds = raw.reshape(4 + num_classes, 8400).T  # (8400, 9)

    cx         = preds[:, 0]
    cy         = preds[:, 1]
    bw         = preds[:, 2]
    bh         = preds[:, 3]
    cls_scores = preds[:, 4:]

    class_ids = np.argmax(cls_scores, axis=1)
    confs     = cls_scores[np.arange(8400), class_ids]

    mask = confs >= conf_thresh
    if not mask.any():
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confs          = confs[mask]
    class_ids      = class_ids[mask]

    x1 = np.clip(((cx - bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y1 = np.clip(((cy - bh / 2) - pad_y) / scale, 0, orig_h).astype(int)
    x2 = np.clip(((cx + bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y2 = np.clip(((cy + bh / 2) - pad_y) / scale, 0, orig_h).astype(int)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices    = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), conf_thresh, iou_thresh)

    detections = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            detections.append((x1[i], y1[i], x2[i], y2[i],
                               float(confs[i]), int(class_ids[i])))
    return detections


# ─── Drawing ─────────────────────────────────────────────────────────────────

PALETTE = [
    (255,  56,  56), 
    (255, 157, 151), 
    (255, 112,  31), 
    (255, 178,  29),
    ( 72, 249,  10),
]


def draw_detections(image, detections, labels):
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = PALETTE[cls_id % len(PALETTE)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = labels[cls_id] if cls_id < len(labels) else f"class {cls_id}"
        text  = f"{label}  {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n TensorRT — Video / Camera Detection")
    parser.add_argument("--engine",      required=True,                    help="Path to .engine file")
    parser.add_argument("--source",      required=True,
                        help="input source: video file path | 0/1/2 (USB camera index) | csi (Jetson CSI camera)")
    parser.add_argument("--output",      default="output.mp4",             help="Output video path (video mode only)")
    parser.add_argument("--conf",        type=float, default=0.5,          help="Confidence threshold")
    parser.add_argument("--iou",         type=float, default=0.45,         help="NMS IoU threshold")
    parser.add_argument("--input-size",  type=int,   default=640,          help="Model input size")
    parser.add_argument("--cam-width",   type=int,   default=640,         help="Camera capture width")
    parser.add_argument("--cam-height",  type=int,   default=480,          help="Camera capture height")
    parser.add_argument("--cam-fps",     type=int,   default=30,           help="Camera FPS")
    parser.add_argument("--no-show",     action="store_true",              help="headless — ไม่เปิด window")
    parser.add_argument("--save",        action="store_true",              help="save output video (camera mode)")
    args = parser.parse_args()

    num-classes=5
    input_size = args.input_size
    labels     = ["person", "bicycle", "car", "motorcycle", "bus"]

    # ── Load engine ──
    print(f"[INFO] Loading engine: {args.engine}")
    engine  = load_engine(args.engine)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    for i, binding in enumerate(engine):
        shape = engine.get_binding_shape(binding)
        kind  = "INPUT" if engine.binding_is_input(binding) else "OUTPUT"
        print(f"[INFO] Binding [{i}] '{binding}'  shape={list(shape)}  {kind}")

    # ── Open source ──
    print(f"[INFO] Opening source: {args.source}")
    cap, is_camera = open_source(args.source, args.cam_width, args.cam_height, args.cam_fps)

    if not cap.isOpened():
        raise IOError(f"can not open source: {args.source}")

    # camera not allow resolution and frame
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or args.cam_fps
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 หรือ 0 สำหรับกล้อง

    mode_str = "camera" if is_camera else "video file"
    print(f"[INFO] {mode_str}: {orig_w}×{orig_h}  {fps:.1f} FPS"
          + (f"  {total} frames" if not is_camera else "  (live)"))

    # ── Letterbox params ──
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    print(f"[INFO] Letterbox: scale={scale:.4f}  pad=({pad_x},{pad_y})")

    # ── VideoWriter ──
    writer   = None
    do_write = (not is_camera) or args.save
    if do_write:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
        print(f"[INFO] Saving output → {args.output}")

    frame_idx = 0
    ms_list   = []
    print("─" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                print("reconnect...")
                cap.release()
                cap, _ = open_source(args.source, args.cam_width, args.cam_height, args.cam_fps)
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] reconnect")
                    break
            else:
                break

        # ── Letterbox ──
        blob, _, _, _ = letterbox_frame(frame, input_size)
        np.copyto(inputs[0]["host"], blob.flatten())

        # ── Inference ──
        t0 = time.perf_counter()
        raw_outputs = infer(context, inputs, outputs, bindings, stream)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000
        ms_list.append(infer_ms)

        # ── Postprocess ──
        detections = postprocess_yolov8(
            raw_outputs[0], orig_w, orig_h,
            scale, pad_x, pad_y,
            num_classes=args.num_classes,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
        )

        # ── Draw ──
        result_frame = draw_detections(frame.copy(), detections, labels)

        cur_fps = 1000.0 / infer_ms
        status = f"{'CAM' if is_camera else 'VID'}  FPS: {cur_fps:.1f}"
        if not is_camera:
            status += f"   {frame_idx}/{total}"
        cv2.putText(result_frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(result_frame)

        if frame_idx % 30 == 0:
            names = [f"{labels[c] if c < len(labels) else f'cls{c}'}({conf:.2f})"
                     for _, _, _, _, conf, c in detections]
            frame_info = f"{frame_idx:5d}/{total}" if not is_camera else f"{frame_idx:5d}/live"
            print(f"[frame {frame_info}]  {infer_ms:5.1f} ms  {cur_fps:4.1f} FPS"
                  f"  → {names if names else 'no detection'}")

        # ── Show window ──
        if not args.no_show:
            cv2.imshow("YOLOv8 TensorRT", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] stopped by user")
                break

        frame_idx += 1

    # ── Cleanup ──
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # ── Summary ──
    avg_ms = sum(ms_list) / len(ms_list) if ms_list else 0
    print("─" * 55)
    print(f"[INFO] finished — {frame_idx} frames")
    print(f"[INFO] Avg inference : {avg_ms:.1f} ms")
    print(f"[INFO] Avg FPS       : {1000/avg_ms:.1f}")
    if do_write:
        print(f"[INFO] Saved => {args.output}")


if __name__ == "__main__":
    main()