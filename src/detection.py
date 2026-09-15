"""
ObjectDetector
==============
YOLO + ByteTrack person detector.

Design rules:
- YOLO model is loaded ONCE in __init__.
- detect() is the only public method; it runs tracking and returns results.
- Weapon detection is intentionally NOT included.
- Every-other-frame skipping (process_every=2) reduces CPU load and keeps
  the WebRTC stream smooth.
"""

from ultralytics import YOLO


class ObjectDetector:
    """YOLO detector for Smart Store Intelligence — person detection only."""

    PERSON_CLASS = 0

    def __init__(self, model_path: str = "yolo26n.pt", process_every: int = 2):
        # YOLO model loads once here — never reloaded during recv().
        self.model = YOLO(model_path)
        self.process_every = max(1, int(process_every))
        self.frame_index: int = 0
        self.last_track_error: str | None = None

    def detect(self, frame):
        """
        Run YOLO + ByteTrack on frame.

        Returns:
            list of Results on inference frames.
            None on skipped frames (caller should use previous frame or
            draw zones on the raw frame and return it).
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        self.frame_index += 1

        # Skip alternate frames to reduce inference load.
        if self.frame_index % self.process_every != 0:
            return None

        try:
            return self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=512,
                conf=0.25,
                classes=[self.PERSON_CLASS],
                verbose=False,
            )
        except ModuleNotFoundError as exc:
            self.last_track_error = (
                "ByteTrack dependency missing: install lap==0.5.13 "
                "in the project venv."
            )
            raise RuntimeError(self.last_track_error) from exc
        except Exception as exc:
            self.last_track_error = f"Tracking failed: {exc}"
            raise
