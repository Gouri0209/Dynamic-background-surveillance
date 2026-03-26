"""
detector.py
-----------
Core object detection engine for dynamic background surveillance.

Techniques used:
  - MOG2 (Mixture of Gaussians) background subtraction
  - Morphological filtering to reduce noise
  - Contour-based object detection
  - Non-Maximum Suppression (NMS) to merge nearby/overlapping boxes
  - Smart labelling: 'Person' vs 'Motion' based on aspect ratio & size
  - Optional Lucas-Kanade Optical Flow tracking
  - HOG + SVM pedestrian detection (OpenCV built-in)
"""

import cv2
import numpy as np


class SurveillanceDetector:
    """
    Detects and tracks objects in video with dynamic/moving backgrounds.

    Parameters
    ----------
    use_optical_flow : bool
        Enable LK Optical Flow for tracking detected objects between frames.
    use_hog : bool
        Enable HOG pedestrian detector as a second-pass classifier.
    bg_history : int
        Number of frames MOG2 uses to build background model.
    var_threshold : float
        MOG2 sensitivity. Lower = more sensitive (more false positives).
    min_contour_area : int
        Ignore contours smaller than this (px²). Filters out tiny noise blobs.
    """

    def __init__(
        self,
        use_optical_flow: bool = True,
        use_hog: bool = False,
        bg_history: int = 500,
        var_threshold: float = 50.0,
        min_contour_area: int = 800,
        nms_threshold: float = 0.3,
    ):
        self.use_optical_flow = use_optical_flow
        self.use_hog = use_hog
        self.min_contour_area = min_contour_area
        self.nms_threshold = nms_threshold  # IoU threshold for merging boxes

        # --- Background Subtractor (MOG2) ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=var_threshold,
            detectShadows=True,   # shadows labelled grey, foreground white
        )

        # --- Morphological kernels ---
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- HOG People Detector ---
        if self.use_hog:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # --- Optical Flow state ---
        self.prev_gray   = None
        self.prev_points = None
        self.lk_params   = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # --- Statistics ---
        self.frame_count    = 0
        self.total_detected = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run the full detection pipeline on a single BGR frame.

        Returns
        -------
        dict with keys:
          annotated  : BGR frame with drawn detections
          fg_mask    : binary foreground mask
          boxes      : list of (x, y, w, h) bounding boxes
          flow_frame : frame with optical flow vectors drawn (or None)
          count      : number of objects detected in this frame
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Background subtraction
        fg_mask = self._apply_bg_subtraction(frame)

        # 2. Find contours → bounding boxes
        boxes = self._find_objects(fg_mask)

        # 2b. Merge overlapping / nearby boxes with NMS
        boxes = self._apply_nms(boxes)

        # 3. Optional HOG pedestrian refinement
        if self.use_hog:
            boxes = self._hog_filter(frame, boxes)

        # 4. Optional Optical Flow tracking
        flow_frame = None
        if self.use_optical_flow:
            flow_frame = self._compute_optical_flow(frame, gray, boxes)

        # 5. Draw annotations
        annotated = self._draw_detections(frame.copy(), boxes)

        # Update stats
        self.total_detected += len(boxes)

        # Prepare next iteration
        self.prev_gray = gray.copy()

        return {
            "annotated":  annotated,
            "fg_mask":    fg_mask,
            "boxes":      boxes,
            "flow_frame": flow_frame,
            "count":      len(boxes),
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

    def _apply_bg_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Apply MOG2 and clean up mask with morphological ops."""
        raw_mask = self.bg_subtractor.apply(frame)

        # Remove shadow pixels (value=127), keep only foreground (value=255)
        _, fg = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        # Remove noise
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open)
        # Fill small holes
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)
        # Slightly expand blobs so they merge
        fg = cv2.dilate(fg, self.kernel_dilate, iterations=2)

        return fg

    def _find_objects(self, mask: np.ndarray) -> list:
        """Extract bounding boxes from foreground mask contours."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio filter: skip very wide/flat blobs (likely background)
            aspect = h / (w + 1e-5)
            if aspect < 0.3 or aspect > 8.0:
                continue
            boxes.append((x, y, w, h))
        return boxes

    def _hog_filter(self, frame: np.ndarray, boxes: list) -> list:
        """
        Run HOG pedestrian detector on ROIs to confirm/reject boxes.
        Keeps a box if HOG also finds a person in that region.
        Falls back to all boxes if HOG finds nothing at all.
        """
        confirmed = []
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        rects, _ = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        if len(rects) == 0:
            return boxes  # no HOG detections → trust MOG2 output

        # Scale HOG rects back to original size
        hog_boxes = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in rects]

        for box in boxes:
            bx, by, bw, bh = box
            for hx, hy, hw, hh in hog_boxes:
                if self._iou(box, (hx, hy, hw, hh)) > 0.1:
                    confirmed.append(box)
                    break
        return confirmed if confirmed else boxes

    def _compute_optical_flow(
        self, frame: np.ndarray, gray: np.ndarray, boxes: list
    ) -> np.ndarray:
        """
        Lucas-Kanade sparse optical flow on feature points inside detections.
        Draws motion vectors on a copy of the frame.
        """
        flow_vis = frame.copy()

        if self.prev_gray is None or len(boxes) == 0:
            self.prev_points = None
            return flow_vis

        # Sample points inside detected bounding boxes
        new_points = []
        for (x, y, w, h) in boxes:
            roi = gray[y:y+h, x:x+w]
            pts = cv2.goodFeaturesToTrack(
                roi, maxCorners=20, qualityLevel=0.3, minDistance=5
            )
            if pts is not None:
                pts[:, 0, 0] += x   # shift to frame coords
                pts[:, 0, 1] += y
                new_points.append(pts)

        if not new_points:
            return flow_vis

        curr_points = np.vstack(new_points).astype(np.float32)

        if self.prev_points is not None and len(self.prev_points) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            if next_pts is not None:
                good_new = next_pts[status == 1]
                good_old = self.prev_points[status == 1]
                for new_pt, old_pt in zip(good_new, good_old):
                    a, b = new_pt.ravel().astype(int)
                    c, d = old_pt.ravel().astype(int)
                    cv2.line(flow_vis, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.circle(flow_vis, (a, b), 3, (0, 0, 255), -1)

        self.prev_points = curr_points
        return flow_vis

    def _apply_nms(self, boxes: list) -> list:
        """
        Non-Maximum Suppression — merges overlapping or nearby boxes.

        Uses a greedy IoU-based approach:
          - Sort boxes by area (largest first)
          - Suppress any smaller box that overlaps the current by > nms_threshold
          - Expand boxes slightly before comparison so nearby (not just overlapping)
            boxes also get merged (proximity merge)
        """
        if len(boxes) <= 1:
            return boxes

        # Sort by area descending
        boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []

        while boxes_sorted:
            current = boxes_sorted.pop(0)
            keep.append(current)
            cx, cy, cw, ch = current
            # Expand current box by 20% for proximity check
            expand = 0.20
            ec = (
                int(cx - cw * expand),
                int(cy - ch * expand),
                int(cw * (1 + 2 * expand)),
                int(ch * (1 + 2 * expand)),
            )
            boxes_sorted = [
                b for b in boxes_sorted
                if self._iou(ec, b) < self.nms_threshold
            ]

        return keep



    def _draw_detections(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Draw bounding boxes labelled as Motion on frame."""
        fh, fw = frame.shape[:2]
        colour = (0, 180, 255)  # orange for all motion detections

        for i, (x, y, w, h) in enumerate(boxes):
            label = f"Object {i + 1}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
            cv2.putText(
                frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # HUD
        cv2.putText(
            frame, f"Detected: {len(boxes)}  |  Frame: {self.frame_count}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _iou(boxA: tuple, boxB: tuple) -> float:
        """Intersection over Union for two (x,y,w,h) boxes."""
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB
        ix = max(ax, bx); iy = max(ay, by)
        iw = min(ax+aw, bx+bw) - ix
        ih = min(ay+ah, by+bh) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = aw*ah + bw*bh - inter
        return inter / union