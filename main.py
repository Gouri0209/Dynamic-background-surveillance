"""
main.py
-------
Entry point for the Smart Surveillance System.

Usage examples
--------------
# Run on webcam (default):
    python main.py

# Run on a video file:
    python main.py --source path/to/video.mp4

# Save output video:
    python main.py --source video.mp4 --save

# Enable HOG pedestrian detector + optical flow:
    python main.py --hog --flow

# Show foreground mask side-by-side:
    python main.py --show-mask

# Run headless (no display) and just save output:
    python main.py --source video.mp4 --save --no-display
"""

import argparse
import sys
import time

import cv2
import numpy as np

from detector import SurveillanceDetector
from yolo_detector import YOLODetector


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smart Surveillance with Dynamic Background Modeling"
    )
    p.add_argument(
        "--source", default=0,
        help="Video source: 0/1 for webcam, or path to a video file (default: 0)"
    )
    p.add_argument("--save",       action="store_true", help="Save annotated output video")
    p.add_argument("--hog",        action="store_true", help="Enable HOG pedestrian detector")
    p.add_argument("--flow",       action="store_true", help="Enable Optical Flow tracking")
    p.add_argument("--show-mask",  action="store_true", help="Show foreground mask panel")
    p.add_argument("--no-display", action="store_true", help="Run headless (no GUI window)")
    p.add_argument("--threshold",  type=float, default=50.0,
                   help="MOG2 sensitivity threshold (default: 50, lower = more sensitive)")
    p.add_argument("--min-area",   type=int, default=800,
                   help="Min contour area in px² to count as object (default: 800)")
    p.add_argument("--output",     default="../output/output.mp4",
                   help="Output video path (default: ../output/output.mp4)")
    p.add_argument("--yolo",       action="store_true",
                   help="Use YOLOv8 instead of MOG2 (requires: pip install ultralytics)")
    p.add_argument("--classes",    nargs="+", default=None,
                   metavar="CLASS",
                   help="Filter YOLO to specific classes e.g. --classes car truck bus person")
    p.add_argument("--yolo-size",  default="n", choices=["n","s","m","l","x"],
                   help="YOLOv8 model size: n=fastest, x=most accurate (default: n)")
    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def open_capture(source) -> cv2.VideoCapture:
    """Open video source; exit gracefully on failure."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)
    return cap


def make_writer(cap: cv2.VideoCapture, path: str) -> cv2.VideoWriter:
    """Create a VideoWriter matching the capture's resolution and FPS."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def build_display(annotated: np.ndarray,
                  fg_mask: np.ndarray,
                  flow_frame,
                  show_mask: bool,
                  use_flow: bool) -> np.ndarray:
    """
    Compose panels side-by-side for the display window.

    Layout:
      [annotated] [mask (opt)] [flow (opt)]
    All panels are resized to the same height.
    """
    panels = [annotated]

    if show_mask:
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, "FG Mask", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
        panels.append(mask_bgr)

    if use_flow and flow_frame is not None:
        flow_copy = flow_frame.copy()
        cv2.putText(flow_copy, "Optical Flow", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
        panels.append(flow_copy)

    if len(panels) == 1:
        return panels[0]

    h = panels[0].shape[0]
    resized = []
    for p in panels:
        scale = h / p.shape[0]
        rw = int(p.shape[1] * scale)
        resized.append(cv2.resize(p, (rw, h)))

    return np.hstack(resized)


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 55)
    print("  Smart Surveillance — Dynamic Background Modeling")
    print("=" * 55)
    print(f"  Source      : {args.source}")
    print(f"  Mode        : {'YOLOv8' if args.yolo else 'MOG2 Motion'}")
    if args.yolo:
        print(f"  YOLO size   : {args.yolo_size}")
        print(f"  Classes     : {args.classes or 'all'}")
    else:
        print(f"  HOG enabled : {args.hog}")
        print(f"  Opt. Flow   : {args.flow}")
        print(f"  Threshold   : {args.threshold}")
        print(f"  Min area    : {args.min_area} px²")
    print(f"  Show mask   : {args.show_mask}")
    print(f"  Save output : {args.save}")
    print("=" * 55)
    print("  Controls: [Q] Quit  [P] Pause  [S] Screenshot")
    print("=" * 55)

    cap = open_capture(args.source)

    if args.yolo:
        detector = YOLODetector(
            model_size=args.yolo_size,
            confidence=0.40,
            filter_classes=args.classes,
        )
    else:
        detector = SurveillanceDetector(
            use_optical_flow=args.flow,
            use_hog=args.hog,
            var_threshold=args.threshold,
            min_contour_area=args.min_area,
        )

    writer = make_writer(cap, args.output) if args.save else None
    paused = False
    screenshot_count = 0

    # FPS tracking
    fps_start = time.time()
    fps_frames = 0
    display_fps = 0.0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream.")
                break

            result = detector.process_frame(frame)

            # FPS calculation
            fps_frames += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            # Stamp FPS onto annotated frame
            cv2.putText(
                result["annotated"],
                f"FPS: {display_fps:.1f}",
                (result["annotated"].shape[1] - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
            )

            display = build_display(
                result["annotated"],
                result["fg_mask"],
                result["flow_frame"],
                args.show_mask,
                args.flow,
            )

            if writer:
                writer.write(result["annotated"])

        # ── Window ──────────────────────────────
        if not args.no_display:
            cv2.imshow("Smart Surveillance System", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n[INFO] Quit by user.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord("s"):
                screenshot_count += 1
                fname = f"../output/screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(fname, result["annotated"])
                print(f"[INFO] Screenshot saved → {fname}")
        else:
            # Headless: just keep processing
            pass

    # ── Cleanup ─────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Output video saved → {args.output}")
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Final stats ─────────────────────────────
    stats = detector.get_stats()
    print("\n── Session Statistics ──────────────────────────")
    print(f"  Frames processed  : {stats['frames_processed']}")
    print(f"  Total detections  : {stats['total_detections']}")
    print(f"  Avg detections/frame: {stats['avg_per_frame']}")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
