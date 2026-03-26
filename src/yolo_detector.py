"""
yolo_detector.py
----------------
YOLOv8-based object detector for specific class detection (cars, people, etc.)

Uses the ultralytics YOLOv8 model pre-trained on COCO (80 classes).
Works on webcam feed or any video file — same interface as the MOG2 detector.

Install:
    pip install ultralytics

Usage (from main.py):
    python main.py --yolo
    python main.py --source road.mp4 --yolo
    python main.py --source road.mp4 --yolo --classes car truck bus
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics is not installed.\n"
        "Run:  pip install ultralytics"
    )


# COCO class colours — consistent colour per class
CLASS_COLOURS = {
    "person":       (0,   220,  60),
    "car":          (0,   180, 255),
    "truck":        (0,   120, 255),
    "bus":          (0,    80, 200),
    "motorcycle":   (255, 180,   0),
    "bicycle":      (255, 220,   0),
    "traffic light":(0,  255, 200),
    "stop sign":    (0,    0,  220),
    "dog":          (220, 100, 255),
    "cat":          (180,  60, 255),
}
DEFAULT_COLOUR = (200, 200, 200)


class YOLODetector:
    """
    Wraps YOLOv8 for real-time object detection with optional class filtering.

    Parameters
    ----------
    model_size : str
        YOLOv8 variant: 'n' (nano, fastest), 's', 'm', 'l', 'x' (most accurate)
        Default 'n' is recommended for real-time webcam use.
    confidence : float
        Minimum confidence threshold (0–1). Default 0.4.
    filter_classes : list[str] | None
        Only show these COCO class names. None = show all 80 classes.
        Example: ['car', 'truck', 'bus', 'person']
    """

    def __init__(
        self,
        model_size: str = "n",
        confidence: float = 0.40,
        filter_classes: list[str] | None = None,
    ):
        model_name = f"yolov8{model_size}.pt"
        print(f"[YOLO] Loading {model_name} — will auto-download if not cached...")
        self.model = YOLO(model_name)
        self.confidence    = confidence
        self.filter_classes = [c.lower() for c in filter_classes] if filter_classes else None

        self.frame_count    = 0
        self.total_detected = 0

    # ------------------------------------------------------------------
    # Public API  (same interface as SurveillanceDetector)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv8 inference on a single BGR frame.

        Returns
        -------
        dict with keys:
          annotated  : BGR frame with drawn detections
          boxes      : list of (x, y, w, h)
          labels     : list of class name strings matching boxes
          count      : number of detections
          fg_mask    : None  (not applicable for YOLO mode)
          flow_frame : None  (not applicable for YOLO mode)
        """
        self.frame_count += 1

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
        )[0]

        boxes  = []
        labels = []

        for box in results.boxes:
            cls_id    = int(box.cls[0])
            cls_name  = self.model.names[cls_id].lower()
            conf_val  = float(box.conf[0])

            # Apply class filter if set
            if self.filter_classes and cls_name not in self.filter_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            boxes.append((x, y, w, h))
            labels.append((cls_name, conf_val))

        self.total_detected += len(boxes)
        annotated = self._draw_detections(frame.copy(), boxes, labels)

        return {
            "annotated":  annotated,
            "boxes":      boxes,
            "labels":     labels,
            "count":      len(boxes),
            "fg_mask":    None,
            "flow_frame": None,
        }

    def get_stats(self) -> dict:
        return {
            "frames_processed": self.frame_count,
            "total_detections": self.total_detected,
            "avg_per_frame":    round(self.total_detected / max(self.frame_count, 1), 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes: list,
        labels: list,
    ) -> np.ndarray:
        """Draw bounding boxes with class name + confidence labels."""

        # Count per class for HUD
        class_counts: dict[str, int] = {}

        for (x, y, w, h), (cls_name, conf_val) in zip(boxes, labels):
            colour = CLASS_COLOURS.get(cls_name, DEFAULT_COLOUR)
            label  = f"{cls_name.title()} {conf_val:.0%}"

            # Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

            # Label pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
            cv2.putText(
                frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # HUD — show per-class counts
        hud_parts = [f"{name.title()}: {cnt}" for name, cnt in class_counts.items()]
        hud = "  |  ".join(hud_parts) if hud_parts else "No detections"
        hud += f"  |  Frame: {self.frame_count}"

        cv2.putText(
            frame, hud,
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 0), 2, cv2.LINE_AA,
        )
        return frame
