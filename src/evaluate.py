"""
evaluate.py
-----------
Offline evaluation utilities for the surveillance system.

Provides:
  - run_on_video()      : process a video file, collect per-frame stats
  - generate_report()   : print + save a plain-text summary report
  - compare_thresholds(): run MOG2 at multiple var_threshold values,
                          report detection counts to help tune sensitivity
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from detector import SurveillanceDetector


# ──────────────────────────────────────────────
# Core evaluation runner
# ──────────────────────────────────────────────

def run_on_video(
    video_path: str,
    detector: SurveillanceDetector,
    save_annotated: bool = False,
    output_path: str = "../output/eval_output.mp4",
    max_frames: int | None = None,
) -> list[dict]:
    """
    Process every frame of a video with the given detector.

    Returns a list of per-frame result dicts:
      { frame_idx, count, processing_ms }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    writer = None
    if save_annotated:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    records = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break

        t0 = time.perf_counter()
        result = detector.process_frame(frame)
        dt = (time.perf_counter() - t0) * 1000  # ms

        records.append({
            "frame_idx": idx,
            "count": result["count"],
            "processing_ms": round(dt, 2),
        })

        if writer:
            writer.write(result["annotated"])

        idx += 1

    cap.release()
    if writer:
        writer.release()

    return records


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_report(
    records: list[dict],
    detector: SurveillanceDetector,
    output_txt: str = "../output/eval_report.txt",
) -> None:
    """Print and save a plain-text evaluation report."""
    counts = [r["count"] for r in records]
    times  = [r["processing_ms"] for r in records]

    lines = [
        "=" * 52,
        "  Surveillance System — Evaluation Report",
        "=" * 52,
        f"  Frames analysed       : {len(records)}",
        f"  Total detections      : {sum(counts)}",
        f"  Max objects / frame   : {max(counts) if counts else 0}",
        f"  Avg objects / frame   : {np.mean(counts):.2f}" if counts else "  Avg objects / frame   : 0",
        f"  Frames with detections: {sum(1 for c in counts if c > 0)}",
        f"  Avg processing time   : {np.mean(times):.1f} ms/frame" if times else "",
        f"  Effective FPS         : {1000/np.mean(times):.1f}" if times else "",
        "-" * 52,
        "  Detector Settings",
        "-" * 52,
        f"  Optical Flow          : {detector.use_optical_flow}",
        f"  HOG Detector          : {detector.use_hog}",
        f"  Min contour area      : {detector.min_contour_area} px²",
        "=" * 52,
    ]

    report = "\n".join(lines)
    print(report)

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w") as f:
        f.write(report + "\n")

    # Also save per-frame CSV
    csv_path = output_txt.replace(".txt", "_frames.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "count", "processing_ms"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[INFO] Report saved  → {output_txt}")
    print(f"[INFO] CSV saved     → {csv_path}")


# ──────────────────────────────────────────────
# Threshold sensitivity comparison
# ──────────────────────────────────────────────

def compare_thresholds(
    video_path: str,
    thresholds: list[float] = [20, 40, 60, 80, 100],
    sample_frames: int = 200,
) -> None:
    """
    Run MOG2 at multiple var_threshold values on the first `sample_frames`
    frames of a video and print a comparison table.

    Helps you pick the right threshold for your scene.
    """
    print("\n── Threshold Sensitivity Analysis ─────────────────")
    print(f"{'Threshold':>12}  {'TotalDet':>9}  {'AvgDet/Frame':>13}  {'AvgMs':>7}")
    print("-" * 50)

    for thresh in thresholds:
        det = SurveillanceDetector(
            use_optical_flow=False,
            use_hog=False,
            var_threshold=thresh,
        )
        records = run_on_video(
            video_path, det,
            save_annotated=False,
            max_frames=sample_frames,
        )
        total   = sum(r["count"] for r in records)
        avg_det = np.mean([r["count"] for r in records]) if records else 0
        avg_ms  = np.mean([r["processing_ms"] for r in records]) if records else 0
        print(f"{thresh:>12.1f}  {total:>9}  {avg_det:>13.2f}  {avg_ms:>7.1f}")

    print("─" * 50)
    print("Tip: Lower threshold → more sensitive (more detections, more noise)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate the surveillance detector on a video.")
    p.add_argument("video", help="Path to input video file")
    p.add_argument("--compare-thresholds", action="store_true",
                   help="Run threshold sensitivity analysis instead of full eval")
    p.add_argument("--save", action="store_true", help="Save annotated output video")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    if args.compare_thresholds:
        compare_thresholds(args.video, sample_frames=args.max_frames or 200)
    else:
        det = SurveillanceDetector(use_optical_flow=True)
        records = run_on_video(
            args.video, det,
            save_annotated=args.save,
            max_frames=args.max_frames,
        )
        generate_report(records, det)
