import cv2


class VideoSourceManager:
    """Optional OpenCV source for webcam/RTSP integrations.

    The current dashboard uses browser WebRTC directly, but this class is
    kept for a conventional OpenCV/CCTV pipeline.
    """

    def __init__(self, source_type="webcam", source=None):
        self.source_type = source_type
        self.source = source
        self.cap = None

    def connect(self):
        if self.source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
        else:
            if not self.source:
                raise ValueError("RTSP source is required.")
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source.")
        return True

    def get_frame(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
