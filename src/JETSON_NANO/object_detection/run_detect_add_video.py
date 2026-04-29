"""
YOLOv8n TensorRT — Video Detection (Letterbox)
Output shape: [1, 9, 8400]  →  9 = 4 (cx,cy,w,h) + 5 classes

Usage:
    python detect_video.py --engine model.engine --video input.mp4 --labels labels.txt
    python detect_video.py --engine model.engine --video input.mp4 --labels labels.txt --no-show
"""

import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  — เริ่ม CUDA context อัตโนมัติ
import time


# ─── TensorRT Logger ─────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# WARNING = แสดงเฉพาะ warning/error ไม่แสดง debug


# ─── Engine Loader ───────────────────────────────────────────────────────────

def load_engine(engine_path: str):
    """โหลด .engine file กลับเป็น TensorRT engine object"""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# ─── Buffer Allocation ───────────────────────────────────────────────────────

def allocate_buffers(engine):
    """จอง pinned CPU memory + GPU memory สำหรับทุก binding"""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()  # คิวคำสั่ง GPU สำหรับ async operations

    for binding in engine:
        size  = trt.volume(engine.get_binding_shape(binding))  # จำนวน element ทั้งหมด
        dtype = trt.nptype(engine.get_binding_dtype(binding))  # FP16 ฯลฯ

        host_mem   = cuda.pagelocked_empty(size, dtype)  # pinned CPU memory
        device_mem = cuda.mem_alloc(host_mem.nbytes)     # GPU memory

        bindings.append(int(device_mem))  # TensorRT ต้องการ GPU address เป็น int

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
        # htod = Host To Device (CPU → GPU), async = ไม่รอให้เสร็จทันที

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # รัน model บน GPU

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        # dtoh = Device To Host (GPU → CPU)

    stream.synchronize()
    # รอให้ทุกคำสั่งใน stream เสร็จจริงๆ ก่อนอ่านผล

    return [out["host"] for out in outputs]


# ─── Letterbox ───────────────────────────────────────────────────────────────

def letterbox_frame(frame_bgr, input_size: int):
    """
    Letterbox: resize รักษา aspect ratio แล้วเติม padding สีเทาสม่ำเสมอ
    ทำให้ภาพไม่บิดเบี้ยว และ model เห็นภาพแบบเดียวกับตอนเทรน

    Returns:
        blob     — numpy CHW float32 [0,1] พร้อมส่ง GPU
        scale    — scale factor ที่ใช้ resize (เดียวกันทั้ง x และ y)
        pad_x    — จำนวน pixel padding ซ้าย
        pad_y    — จำนวน pixel padding บน
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    # หา scale ที่เล็กกว่า เพื่อให้ภาพพอดีโดยไม่ overflow
    # ตัวอย่าง: ภาพ 1280×720 → scale = min(640/1280, 640/720) = min(0.5, 0.889) = 0.5
    scale = min(input_size / orig_w, input_size / orig_h)

    new_w = int(orig_w * scale)  # ความกว้างหลัง resize
    new_h = int(orig_h * scale)  # ความสูงหลัง resize

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # INTER_LINEAR = bilinear interpolation คุณภาพดีพอ เร็ว

    # สร้าง canvas สีเทา 640×640
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    # 114 = ค่าสีเทา standard ที่ YOLO ใช้เป็น padding

    # คำนวณตำแหน่งที่จะวางภาพให้อยู่กลาง canvas
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    # วางภาพลงบน canvas
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    return np.ascontiguousarray(blob), scale, pad_x, pad_y


# ─── Postprocessing ──────────────────────────────────────────────────────────

def postprocess_yolov8(raw, orig_w, orig_h, scale, pad_x, pad_y,
                       num_classes=5, conf_thresh=0.5, iou_thresh=0.45):
    """
    แปลง raw output [1, 9, 8400] → list of (x1, y1, x2, y2, conf, class_id)
    coordinates อยู่ใน original image space แล้ว

    YOLOv8 output layout (ต่างจาก YOLOv5):
      row 0-3 : cx, cy, w, h  (ใน 640×640 letterbox space)
      row 4-8 : class scores  (ไม่มี objectness score)
    """
    # raw มาเป็น flat array → reshape (9, 8400) → transpose (8400, 9)
    # แต่ละแถวคือ 1 prediction box
    preds = raw.reshape(4 + num_classes, 8400).T  # (8400, 9)

    cx         = preds[:, 0]   # center x ของทุก box ใน letterbox space
    cy         = preds[:, 1]   # center y
    bw         = preds[:, 2]   # ความกว้าง box
    bh         = preds[:, 3]   # ความสูง box
    cls_scores = preds[:, 4:]  # (8400, 5) — score ของแต่ละ class

    # หา class ที่มี score สูงสุดสำหรับแต่ละ prediction
    class_ids = np.argmax(cls_scores, axis=1)          # (8400,)
    confs     = cls_scores[np.arange(8400), class_ids] # (8400,) ดึง score ของ class ที่ชนะ

    # กรองเอาแค่ prediction ที่ confidence >= threshold
    mask = confs >= conf_thresh
    if not mask.any():
        return []  # ไม่มี detection เลย

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confs          = confs[mask]
    class_ids      = class_ids[mask]

    # แปลง letterbox coords → original image coords
    # ขั้นตอน: ลบ pad ออกก่อน (เพราะ letterbox เพิ่ม pad) แล้วค่อย หาร scale
    #
    #   letterbox space:  cx, cy อยู่ใน 640×640 รวม padding
    #   หลังลบ pad:       cx - pad_x = ตำแหน่งใน area ที่มีภาพจริง
    #   หลังหาร scale:    ตำแหน่งใน original image
    #
    x1 = np.clip(((cx - bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y1 = np.clip(((cy - bh / 2) - pad_y) / scale, 0, orig_h).astype(int)
    x2 = np.clip(((cx + bw / 2) - pad_x) / scale, 0, orig_w).astype(int)
    y2 = np.clip(((cy + bh / 2) - pad_y) / scale, 0, orig_h).astype(int)
    # np.clip ป้องกัน coordinate ออกนอกขอบภาพ

    # NMS — ลบ box ที่ซ้อนทับกันเกิน iou_thresh ออก
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    # cv2.dnn.NMSBoxes ต้องการ format [x, y, w, h]

    indices = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), conf_thresh, iou_thresh)
    # คืน index ของ box ที่รอดจาก NMS

    detections = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            detections.append((
                x1[i], y1[i], x2[i], y2[i],
                float(confs[i]),
                int(class_ids[i])
            ))
    return detections


# ─── Drawing ─────────────────────────────────────────────────────────────────
#color to drawing square frame
PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    ( 72, 249,  10)
]


def draw_detections(image, detections, labels=None):
    """วาด bounding box + label + confidence บนภาพ"""
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = PALETTE[cls_id % len(PALETTE)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # วาดกรอบ — 2 คือความหนาเส้น pixel

        label = labels[cls_id] if labels and cls_id < len(labels) else f"class {cls_id}"
        text  = f"{label}  {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # วัดขนาดข้อความก่อน เพื่อวาด background ป้ายให้พอดี

        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        # วาดสี่เหลี่ยมทึบ (-1) เป็น background ป้ายชื่อ

        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # เขียนชื่อ class + confidence สีขาว, LINE_AA = anti-aliasing

    return image


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n TensorRT — Video Detection")
    parser.add_argument("--engine",      required=True,          help="Path to .engine file")
    parser.add_argument("--video",       required=True,          help="Path to input video file")
    parser.add_argument("--output",      default="first_video_output.mp4",   help="Output video path")
    parser.add_argument("--conf",        type=float, default=0.5,  help="Confidence threshold")
    parser.add_argument("--iou",         type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--num-classes", type=int,   default=5,    help="Number of classes")
    parser.add_argument("--input-size",  type=int,   default=640,  help="Model input size")
    parser.add_argument("--no-show",     action="store_true",      help="headless — ไม่เปิด window")
    args = parser.parse_args()

    input_size = args.input_size  # 640

    # ── Load labels ──
    labels = ["person","bicycle","car","motorcycle","bus"]

    # ── Load engine ──
    print(f"[INFO] Loading engine: {args.engine}")
    engine  = load_engine(args.engine)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Debug — print binding shapes เพื่อ verify
    for i, binding in enumerate(engine):
        shape = engine.get_binding_shape(binding)
        kind  = "INPUT" if engine.binding_is_input(binding) else "OUTPUT"
        print(f"[INFO] Binding [{i}] '{binding}'  shape={list(shape)}  {kind}")

    # ── Open video ──
    print(f"[INFO] Video: {args.video}")
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise IOError(f"can not open video: {args.video}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # ความกว้าง frame ต้นฉบับ
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # ความสูง frame ต้นฉบับ
    fps    = cap.get(cv2.CAP_PROP_FPS)                # FPS ของวิดีโอ
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # จำนวน frame ทั้งหมด

    print(f"[INFO] {orig_w}×{orig_h}  {fps:.1f} FPS  {total} frames")

    # ── Prepare letterbox params ──
    # คำนวณ scale + pad ครั้งเดียว เพราะทุก frame มีขนาดเท่ากัน
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    print(f"[INFO] Letterbox: scale={scale:.4f}  pad=({pad_x},{pad_y})")

    # ── Prepare VideoWriter ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec สำหรับ .mp4
    writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
    # output video มีขนาดเท่ากับ input (orig_w × orig_h)

    frame_idx = 0   # นับ frame
    ms_list   = []  # เก็บ inference time ทุก frame สำหรับคำนวณ avg

    print("[INFO] กด 'q' เพื่อหยุด")
    print("─" * 55)

    while True:
        ret, frame = cap.read()

        if not ret:
            break 

        # ── Letterbox ──
        blob, _, _, _ = letterbox_frame(frame, input_size)
        # scale, pad_x, pad_y คำนวณไว้แล้วข้างบน ไม่ต้องรับซ้ำ
        # (frame ทุก frame มีขนาดเท่ากัน)

        np.copyto(inputs[0]["host"], blob.flatten())
        # copy blob เข้า pinned CPU buffer ก่อน infer

        # ── Inference ──
        t0 = time.perf_counter()
        raw_outputs = infer(context, inputs, outputs, bindings, stream)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000  # แปลงเป็น millisecond
        ms_list.append(infer_ms)

        # ── Postprocess ──
        detections = postprocess_yolov8(
            raw_outputs[0],
            orig_w, orig_h,
            scale, pad_x, pad_y,           # letterbox params
            num_classes=args.num_classes,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
        )

        # ── Draw ──
        result_frame = draw_detections(frame.copy(), detections, labels)

        # วาด FPS + frame counter มุมบนซ้าย
        cur_fps = 1000.0 / infer_ms
        cv2.putText(result_frame,
                    f"FPS: {cur_fps:.1f}   frame: {frame_idx}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # ── Save frame ──
        writer.write(result_frame)

        # ── Print ทุก 30 frame ──
        if frame_idx % 30 == 0:
            names = []
            for _, _, _, _, conf, cls_id in detections:
                n = labels[cls_id] if labels and cls_id < len(labels) else f"class_{cls_id}"
                names.append(f"{n}({conf:.2f})")
            print(f"[frame {frame_idx:5d}/{total}]  {infer_ms:5.1f} ms  "
                  f"{cur_fps:4.1f} FPS  → {names if names else 'no detection'}")

        # ── Show window ──
        if not args.no_show:
            cv2.imshow("YOLOv8 TensorRT — Video", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] stoped")
                break

        frame_idx += 1

    # ── Cleanup ──
    cap.release()      # ปิด VideoCapture คืน resource
    writer.release()   # flush + ปิดไฟล์วิดีโอ output
    cv2.destroyAllWindows()

    # ── Summary ──
    avg_ms = sum(ms_list) / len(ms_list) if ms_list else 0
    print("─" * 55)
    print(f"[INFO] finished — {frame_idx} frames")
    print(f"[INFO] Avg inference : {avg_ms:.1f} ms")
    print(f"[INFO] Avg FPS       : {1000/avg_ms:.1f}")
    print(f"[INFO] Saved         → {args.output}")


if __name__ == "__main__":
    main()