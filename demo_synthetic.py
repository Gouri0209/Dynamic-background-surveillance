"""
demo_synthetic.py
-----------------
Generates a synthetic surveillance scene (no camera or video file needed)
and runs the detector on it — useful for quick testing and demos.

The synthetic scene contains:
  - A slowly drifting background (simulates dynamic background like swaying)
  - Multiple "objects" (filled rectangles) that move across the frame
  - Random Gaussian noise on every frame

Run:
    python demo_synthetic.py
    python demo_synthetic.py --save
    python demo_synthetic.py --frames 300 --objects 4
"""

import argparse
import time

import cv2
import numpy as np

from detector import SurveillanceDetector


# ──────────────────────────────────────────────
# Synthetic scene generator
# ──────────────────────────────────────────────

class SyntheticScene:
    """Procedurally generates surveillance-like frames with moving objects."""

    def __init__(self, width=640, height=480, num_objects=3, seed=42):
        self.W = width
        self.H = height
        rng = np.random.default_rng(seed)

        # Static background: gradient + subtle texture
        self.bg = self._make_background(rng)

        # Moving objects: each is (x, y, w, h, vx, vy, colour)
        self.objects = []
        colours = [(200, 80, 60), (60, 180, 100), (80, 100, 220)]
        for i in range(num_objects):
            ox = int(rng.integers(50, width - 100))
            oy = int(rng.integers(50, height - 80))
            ow = int(rng.integers(40, 90))
            oh = int(rng.integers(60, 110))
            vx = float(rng.choice([-1, 1])) * rng.uniform(1.5, 3.5)
            vy = float(rng.choice([-1, 1])) * rng.uniform(0.5, 2.0)
            col = colours[i % len(colours)]
            self.objects.append([ox, oy, ow, oh, vx, vy, col])

        self.frame_idx = 0

    def _make_background(self, rng) -> np.ndarray:
        bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # Gradient sky
        for row in range(self.H):
            val = int(80 + (row / self.H) * 60)
            bg[row, :] = (val, val + 10, val + 20)
        # Noisy texture overlay
        noise = rng.integers(0, 12, (self.H, self.W, 3), dtype=np.uint8)
        bg = cv2.add(bg, noise)
        return bg

    def next_frame(self) -> np.ndarray:
        """Render the next frame."""
        # Dynamic background: slow sine-wave brightness shift
        t = self.frame_idx / 30.0
        shift = int(8 * np.sin(t))
        frame = np.clip(self.bg.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        # Per-frame noise (simulates sensor noise / dynamic texture)
        noise = np.random.randint(-6, 6, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Move and draw objects
        for obj in self.objects:
            ox, oy, ow, oh, vx, vy, col = obj
            ox += vx; oy += vy
            # Bounce at edges
            if ox < 0 or ox + ow > self.W:
                vx = -vx; ox = max(0, min(ox, self.W - ow))
            if oy < 0 or oy + oh > self.H:
                vy = -vy; oy = max(0, min(oy, self.H - oh))
            # Draw filled rectangle (the "object")
            cv2.rectangle(frame, (int(ox), int(oy)),
                          (int(ox+ow), int(oy+oh)), col, -1)
            # Slight shadow
            cv2.rectangle(frame, (int(ox)+3, int(oy)+3),
                          (int(ox+ow)+3, int(oy+oh)+3), (20, 20, 20), 2)
            # Save updated state
            obj[0], obj[1], obj[4], obj[5] = ox, oy, vx, vy

        # Overlay frame counter
        cv2.putText(frame, f"Synthetic Frame {self.frame_idx}",
                    (8, self.H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        self.frame_idx += 1
        return frame


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run detector on a synthetic scene.")
    p.add_argument("--frames",  type=int, default=400, help="Number of frames to render")
    p.add_argument("--objects", type=int, default=3,   help="Number of moving objects")
    p.add_argument("--save",    action="store_true",   help="Save output to ../output/demo.mp4")
    p.add_argument("--flow",    action="store_true",   help="Show optical flow panel")
    p.add_argument("--mask",    action="store_true",   help="Show foreground mask panel")
    args = p.parse_args()

    scene    = SyntheticScene(num_objects=args.objects)
    detector = SurveillanceDetector(
        use_optical_flow=args.flow,
        use_hog=False,
        var_threshold=40,
        min_contour_area=600,
    )

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("../output/demo.mp4", fourcc, 25, (640, 480))

    print(f"[INFO] Running synthetic demo ({args.frames} frames) …")
    print("[INFO] Press Q to quit, P to pause, S for screenshot.")

    paused = False
    screenshots = 0

    for _ in range(args.frames):
        raw = scene.next_frame()

        if not paused:
            result = detector.process_frame(raw)

            panels = [result["annotated"]]
            if args.mask:
                panels.append(cv2.cvtColor(result["fg_mask"], cv2.COLOR_GRAY2BGR))
            if args.flow and result["flow_frame"] is not None:
                panels.append(result["flow_frame"])

            display = np.hstack(panels) if len(panels) > 1 else panels[0]

            if writer:
                writer.write(result["annotated"])
        
        cv2.imshow("Synthetic Demo — Smart Surveillance", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            screenshots += 1
            cv2.imwrite(f"../output/demo_shot_{screenshots:03d}.jpg", result["annotated"])
            print(f"[INFO] Screenshot → demo_shot_{screenshots:03d}.jpg")

    if writer:
        writer.release()
        print("[INFO] Demo video saved → ../output/demo.mp4")

    cv2.destroyAllWindows()
    stats = detector.get_stats()
    print(f"\n[DONE] Frames: {stats['frames_processed']} | "
          f"Total detections: {stats['total_detections']} | "
          f"Avg/frame: {stats['avg_per_frame']}")


if __name__ == "__main__":
    main()
