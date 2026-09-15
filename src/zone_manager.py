"""
ZoneManager — zone definitions and drawing helpers
===================================================

Supports camera-specific zone configurations.

Default zones (camera_01 / backward-compatible):
  Entrance, Shelf A, Checkout

To use camera-specific zones, pass a zones dict to __init__:
  ZoneManager(zones={
      "Entrance": {"coords": (50, 260, 220, 360), "color": (0,255,0)},
      ...
  })

The class-level DEFAULT_ZONES dict acts as the fallback for
camera_01 (single-camera mode) and as a reference configuration
template for future cameras.
"""

from __future__ import annotations

import cv2
from typing import Optional


class ZoneManager:
    # Default zone layout for camera_01 (Entrance → Shelf A → Checkout).
    # Future cameras supply their own zones at construction time.
    DEFAULT_ZONES: dict = {
        "Entrance": {"coords": (50,  260, 220, 360), "color": (0, 255, 0)},
        "Shelf A":  {"coords": (220, 260, 400, 360), "color": (255, 0, 0)},
        "Checkout": {"coords": (400, 260, 590, 360), "color": (0, 0, 255)},
    }

    # Per-camera default configurations.
    # Add entries here as new cameras are added.
    CAMERA_ZONES: dict = {
        "camera_01": {
            "Entrance": {"coords": (50,  260, 220, 360), "color": (0, 255, 0)},
            "Shelf A":  {"coords": (220, 260, 400, 360), "color": (255, 0, 0)},
            "Checkout": {"coords": (400, 260, 590, 360), "color": (0, 0, 255)},
        },
        # Future cameras — add their zone layouts here:
        # "camera_02": { "Shelf B": { ... }, "Shelf C": { ... } },
        # "camera_03": { "Checkout A": { ... } },
        # "camera_04": { "Exit": { ... } },
    }

    def __init__(
        self,
        camera_id: str = "camera_01",
        zones: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        camera_id : str
            Camera identifier.  Used to look up a default zone config
            from CAMERA_ZONES when zones=None.
        zones : dict | None
            Explicit zone config (overrides CAMERA_ZONES lookup).
            Format: { zone_name: { "coords": (x1,y1,x2,y2), "color": (B,G,R) } }
        """
        self.camera_id = camera_id
        if zones is not None:
            self.ZONES = zones
        else:
            self.ZONES = self.CAMERA_ZONES.get(camera_id, self.DEFAULT_ZONES)

    def get_zone(self, x: int, y: int) -> Optional[str]:
        for name, zone in self.ZONES.items():
            x1, y1, x2, y2 = zone["coords"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def draw_zones(self, frame):
        for name, zone in self.ZONES.items():
            x1, y1, x2, y2 = zone["coords"]
            color = zone["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1 + 5, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def draw_person_zone(
        self,
        frame,
        feet_x: int,
        feet_y: int,
        zone_name: Optional[str],
        track_id=None,
    ):
        label = f"Zone: {zone_name}" if zone_name else "Zone: Unknown"
        if track_id is not None:
            label = f"ID:{track_id} | {label}"
        cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 255, 255), -1)
        cv2.putText(frame, label, (int(feet_x) + 8, int(feet_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame
