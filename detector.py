"""detector -- YOLO11n ONNX person detection for camwatch.

CPU-first: the session always starts on CPU (the GPU stays free).
Only after a FIRST CONFIRMED person exactly ONE GPU switch probe may run
(maybe_gpu_once); if it fails, the process stays on CPU forever -- both by design, no crash.

All coordinates in the returned dict are PIXELS of the ORIGINAL camera image (e.g. 640x480).
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image, ImageDraw

try:  # camwatch supplies Config + Logging when the library sits next to this file
    from camwatch_lib import CFG as _CFG, log as _log
except Exception:      # standalone import (tests): gentle fallback
    _CFG = {"cpu_threads": 2, "conf_threshold": 0.35, "iou_threshold": 0.45}

    def _log(msg): pass


COCO_NAMES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign",
              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
              "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
              "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "shears", "teddy bear",
              "hair drier", "toothbrush")

N_CLASSES = len(COCO_NAMES)          # 80
PERSON_IDX = COCO_NAMES.index("person")   # 0 -- computed explicitly, never hard-coded


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two boxes (x1,y1,x2,y2)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - iw*ih
    return (iw*ih) / ua if ua > 1e-9 else 0.0


class Detector:
    """One instance per process (the daemon); detect() shares its session."""

    IM = 640   # fixed input size of the ONNX export -- all dimensions derive from it

    def __init__(self, model_path=None, conf_threshold=None, iou_threshold=None):
        import onnxruntime as ort               # local import: cached in Python afterwards and cheap
        self.ort = ort
        path   = str(model_path or _CFG["model"])      # CFG holds the absolute path after lib's _fix()
        opts   = ort.SessionOptions()
        opts.intra_op_num_threads = int(_CFG.get("cpu_threads", 2))
        self.sess       = ort.InferenceSession(path, options=opts, providers=["CPUExecutionProvider"])
        self.model_path = path
        self.conf    = float(conf_threshold if conf_threshold is not None else _CFG.get("conf_threshold", 0.35))
        self.iou_thr = float(iou_threshold if iou_threshold is not None else _CFG.get("iou_threshold", 0.45))
        self.gpu           = False                # currently running on GPU? only after a maybe_gpu_once() success
        self._gpu_decided = False                 # guarantees at most ONE switch attempt per process (success or failure)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    # ------------------------------------------------------------ inference ----------------------------------------

    def _preprocess(self, img: Image.Image):
        """PIL-RGB -> (NCHW float32/255 blob 640^2, geo). geo=(w,h,scale,px,py) maps boxes back to the original image."""
        w, h = img.size
        s     = self.IM / max(w, h)
        nw, nh = int(round(w * s)), int(round(h * s))
        arr   = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        canvas = np.full((self.IM, self.IM, 3), 114 / 255.0, dtype=np.float32)    # YOLO padding grey
        px, py = (self.IM - nw) // 2, (self.IM - nh) // 2
        canvas[py:py+nh, px:px+nw] = arr
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)     # NCHW as the YOLO export expects
        return blob, (float(w), float(h), s, float(px), float(py))

    def _decode(self, raw: np.ndarray, geo):
        """Raw output ([N,C] or [C,N], C = 4+80) -> NMS-filtered boxes in original-image pixels."""
        w, h, s, px, py = geo
        a = np.squeeze(raw)                                       # [C,N] or [N,C]
        if a.ndim != 2:
            raise RuntimeError(f"unexpected model output shape {a.shape}")
        n_cols = 4 + N_CLASSES                                     # expect 84 (SOFTMAX is baked into the export!)
        m = a.T if a.shape[0] == n_cols else a                    # normalized -> rows [cx,cy,w,h, c_0 ... c_79]
        box, scores = m[:, :4].astype(np.float32), m[:, 4:].astype(np.float32)
        maxc   = scores.max(axis=1)
        cid    = scores.argmax(axis=1)                            # COCO index directly -- no extra argmax race needed
        top_ids = np.where(maxc >= self.conf)[0]                  # confidence filter BEFORE NMS (cheap + stable!)

        keep: list[list[float]] = []                              # entries: [conf, x1, y1, x2, y2, class_idx]
        for i in sorted(top_ids.tolist(), key=lambda k: float(maxc[k]), reverse=True)[:64]:
            if float(maxc[i]) < self.conf:                       # defensive re-check after sorting
                continue
            cx, cy, bw_, bh_ = (float(v) for v in box[i])
            x1 = max(0.0, ((cx - bw_/2) - px) / s);  y1 = max(0.0, ((cy - bh_/2) - py) / s)
            x2 = min(w,   ((cx + bw_/2) - px) / s);  y2 = min(h,   ((cy + bh_/2) - py) / s)
            if (x2 - x1) * (y2 - y1) < 4.0:                      # smaller than 2x2 px -- noise, dropped
                continue
            b_new = [float(maxc[i]), float(x1), float(y1), float(x2), float(y2), int(cid[i])]
            if any(_iou([cn[1], cn[2], cn[3], cn[4]], [b_new[1], b_new[2], b_new[3], b_new[4]]) >= self.iou_thr for cn in keep):
                continue                                        # lower-confidence same-class box -> rejected (greedy)
            keep.append(b_new)

        persons = [{"x1": d[1], "y1": d[2], "x2": d[3], "y2": d[4], "conf": round(float(d[0]), 3)} for d in keep if int(d[5]) == PERSON_IDX]
        labels  = sorted(((COCO_NAMES[int(c[5])], round(float(c[0]), 3)) for c in keep if int(c[5]) != PERSON_IDX),
                         key=lambda z: -z[1])[:5]                # top-5 remaining classes for the info banner
        return persons, labels

    def detect(self, jpeg_bytes: bytes) -> dict:
        """-> {'w','h','ms','persons':[{'x1','y1','x2','y2','conf'}], 'labels':[(name,conf),...]}"""
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        blob, geo = self._preprocess(img)
        t0  = time.monotonic()
        raw = self.sess.run([self.out_name], {self.in_name: blob})[0]
        ms  = (time.monotonic() - t0) * 1000.0
        persons, labels = self._decode(raw, geo)
        return {"w": img.size[0], "h": img.size[1], "ms": round(ms, 1),
                "persons": persons, "labels": labels}

    # --------------------------------------------------- GPU switch: exactly ONCE ------------------------------

    def maybe_gpu_once(self) -> bool:
        """A single CUDA attempt; from then on the decision is final for this process."""
        if self._gpu_decided:
            return False
        self._gpu_decided = True            # BEFORE the probe -> a crash during the switch never triggers a 2nd attempt
        try:
            opts = self.sess.get_session_options() or self.ort.SessionOptions()
            gpu_sess = self.ort.InferenceSession(          # same graph, only CUDA listed before CPU (CPU remains fallback)
                self.model_path, options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            actives = [type(p).__name__ for p in gpu_sess.get_providers()]
            if not any("CUDA" in name for name in actives):     # session success != GPU active! This single line is the honest check.
                _log("[gpu] session runs on CPU only despite CUDA provider list (providers: %s) -- continuing with CPU." % ", ".join(actives))
                return False
            self.in_name  = gpu_sess.get_inputs()[0].name
            self.out_name = gpu_sess.get_outputs()[0].name
            self.sess     = gpu_sess                   # from now on all inferences of this process run on GPU; pre-processing/decode unchanged
            self.gpu      = True
            _log("[gpu] CUDA switch successful -- all further inferences in this process run on the GPU (providers: %s)." % ", ".join(actives))
            return True
        except Exception as e:                # CUDA EP missing in build / driver absent -> known case, log it, CPU from now on forever
            _log("[gpu] CUDA attempt failed (%s: %s) -- staying on CPU permanently (as designed)." % (type(e).__name__, str(e)[:120]))
            return False


# ========================================================== annotation =========================================

def annotate_frame(jpeg_bytes: bytes, result: dict) -> bytes:
    """Green boxes around persons + red banner 'PERSON DETECTED <time>'; returns JPEG bytes."""
    img  = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)                # PIL has no get_font() without FreeType -> default bitmap font, clear enough for the banner.
    now  = time.strftime("%d.%m. %H:%M:%S")

    def text_w(text: str) -> int:
        return int(draw.textlength(text)) if hasattr(draw, "textlength") else len(text) * 7   # fallback math for old PIL versions

    persons = result.get("persons") or []
    for p in persons:                                # green boxes + label with confidence below/above the box
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        draw.rectangle([max(0, x1-3), max(0, y1-3), min(img.size[0], x2+3), min(img.size[1], y2+3)], outline=(0, 235, 60), width=4)
        label = f"Person {p['conf']:.2f}"
        ly    = max(0, y1 - 28)                     # label box above the detection; near the top edge it is simply clipped at 0.
        lw    = text_w(label) + 14
        draw.rectangle([x1-3, ly, x1 - 3 + max(lw, int(text_w("xx"))+2), ly + 24], fill=(0, 185, 45))
        draw.text((x1 + 4, ly + 3), label, fill=(255, 255, 255))

    banner = "PERSON DETECTED   %s" % now             # red on top; only produced if result['persons'] is non-empty (caller controls sending!)
    bw     = text_w(banner) + 24
    draw.rectangle([0, 0, bw, 38], fill=(205, 15, 25))
    draw.text((10, 6), banner, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)         # clear for Telegram preview AND zoom; compact enough for archiving.
    return buf.getvalue()
