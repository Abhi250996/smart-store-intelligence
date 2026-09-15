import numpy as np
import av

from dashboard import SmartStoreVideoProcessor
from src.event_engine import EventEngine


def test_first_zone_entry():
    engine = EventEngine()
    zone, dwell, event = engine.update_dwell_time(1, "Entrance")
    assert zone == "Entrance"
    assert dwell == 0.0
    assert event["event_type"] == "ZONE_ENTRY"


def test_same_zone_does_not_create_transition():
    engine = EventEngine()
    engine.update_dwell_time(1, "Shelf A")
    zone, _, event = engine.update_dwell_time(1, "Shelf A")
    assert zone == "Shelf A"
    assert event is None


def test_pending_counts_are_initialized_for_new_zone_transition_cycle():
    engine = EventEngine()
    engine.active_tracks[1] = {"zone": "Entrance", "entry_time": 0.0}
    engine.pending_zones[1] = "Shelf A"
    engine.pending_counts.pop(1, None)

    # A missing pending counter should be lazily initialized instead of
    # raising KeyError while checking whether the zone change becomes stable.
    zone, dwell, event = engine.update_dwell_time(1, "Shelf A")

    assert zone == "Entrance"
    assert dwell >= 0.0
    assert event is None


def test_video_processor_handles_new_track_without_detections_and_missing_ids():
    processor = SmartStoreVideoProcessor()
    processor.detector.detect = lambda img: []
    processor.weapon_security_monitor.model = None

    class DummyFrame:
        def to_ndarray(self, format="bgr24"):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    frame = DummyFrame()
    output = processor.recv(frame)

    assert isinstance(output, av.VideoFrame)

    # Simulate person detection without a track ID and a weapon detection
    # without a track ID; these are recoverable per-frame states.
    processor.detector.detect = lambda img: [
        type(
            "Result",
            (),
            {
                "boxes": type(
                    "Boxes",
                    (),
                    {
                        "xyxy": type("XYXY", (), {"cpu": lambda self: type("X", (), {"numpy": lambda self: np.array([[10, 10, 50, 80], [60, 40, 100, 100]], dtype=float)})()})(),
                        "cls": type("CLS", (), {"int": lambda self: type("I", (), {"cpu": lambda self: type("C", (), {"tolist": lambda self: [0, 0]})()})()})(),
                        "conf": type("CONF", (), {"cpu": lambda self: type("C", (), {"tolist": lambda self: [0.9, 0.8]})()})(),
                        "id": type("ID", (), {"int": lambda self: type("I", (), {"cpu": lambda self: type("C", (), {"tolist": lambda self: [None, None]})()})()})(),
                    },
                )(),
                "names": {0: "person"},
                "plot": lambda self: np.zeros((240, 320, 3), dtype=np.uint8),
            }
        )()
    ]
    output = processor.recv(frame)
    assert isinstance(output, av.VideoFrame)
