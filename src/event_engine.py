"""
EventEngine — in-memory tracking state and event generation
============================================================

Accepts camera_id at construction.  Every generated event dict
now includes "camera_id" so EventStore can persist it.

Existing callers that don't pass camera_id receive "camera_01".
All tracking logic is unchanged.
"""

from __future__ import annotations

import time


class EventEngine:

    def __init__(self, camera_id: str = "camera_01") -> None:

        self.camera_id = camera_id          # stamped on every event dict

        self.active_tracks: dict = {}
        self.triggered_events: dict = {}
        self.missing_counts: dict = {}
        self.zone_history: dict = {}
        self.cumulative_dwell_by_zone: dict = {}
        self.completed_tracks: set = set()

        self.active_customer_ids: dict = {}
        self.customer_counter: int = 0

        self.pending_zones: dict = {}
        self.pending_counts: dict = {}

        self.ZONE_STABILITY_FRAMES: int = 5

        self.session_id: int = int(time.time() * 1000) % 1_000_000

    # ── helpers ───────────────────────────────────────────────────────────

    def _ev(self, d: dict) -> dict:
        """Stamp camera_id onto every event dict before returning it."""
        d["camera_id"] = self.camera_id
        return d

    # ── customer ID ───────────────────────────────────────────────────────

    def _create_customer_id(self, track_id):
        self.customer_counter += 1
        customer_id = self.session_id * 1_000_000 + self.customer_counter
        self.active_customer_ids[track_id] = customer_id
        return customer_id

    def get_customer_id(self, track_id):
        return self.active_customer_ids.get(track_id)

    def _get_or_create_customer_id(self, track_id):
        if track_id not in self.active_customer_ids:
            return self._create_customer_id(track_id)
        return self.active_customer_ids[track_id]

    # ── dwell tracking ────────────────────────────────────────────────────

    def update_dwell_time(self, track_id, current_zone):
        current_time = time.time()
        customer_id  = self._get_or_create_customer_id(track_id)

        self.completed_tracks.discard(track_id)
        self.missing_counts[track_id] = 0

        # ── no zone ──────────────────────────────────────────────────────
        if current_zone is None:
            if track_id not in self.active_tracks:
                if track_id in self.completed_tracks:
                    return None, 0.0, None

                self.active_tracks[track_id] = {
                    "zone": current_zone, "entry_time": current_time
                }
                self.cumulative_dwell_by_zone.setdefault(track_id, {})
                self.completed_tracks.discard(track_id)
                self.missing_counts[track_id] = 0
                self.zone_history[track_id] = [current_zone]
                return current_zone, 0.0, self._ev({
                    "event_type": "ZONE_ENTRY",
                    "track_id":   customer_id,
                    "zone_name":  current_zone,
                })

        # ── new track ─────────────────────────────────────────────────────
        if track_id not in self.active_tracks:
            customer_id = self._get_or_create_customer_id(track_id)
            self.active_tracks[track_id] = {
                "zone": current_zone, "entry_time": current_time
            }
            self.cumulative_dwell_by_zone.setdefault(track_id, {})
            self.missing_counts[track_id] = 0
            self.zone_history[track_id] = [current_zone]
            return current_zone, 0.0, self._ev({
                "event_type": "ZONE_ENTRY",
                "track_id":   customer_id,
                "zone_name":  current_zone,
            })

        customer_id   = self._get_or_create_customer_id(track_id)
        previous_zone = self.active_tracks[track_id]["zone"]
        entry_time    = self.active_tracks[track_id]["entry_time"]
        dwell_time    = current_time - entry_time

        # ── same zone ─────────────────────────────────────────────────────
        if previous_zone == current_zone:
            self.pending_zones.pop(track_id, None)
            self.pending_counts.pop(track_id, None)
            return current_zone, round(dwell_time, 1), None

        # ── possible zone change ──────────────────────────────────────────
        if self.pending_zones.get(track_id) != current_zone:
            self.pending_zones[track_id]  = current_zone
            self.pending_counts[track_id] = 1
            return previous_zone, round(dwell_time, 1), None

        self.pending_counts[track_id] = self.pending_counts.get(track_id, 0) + 1

        if self.pending_counts[track_id] < self.ZONE_STABILITY_FRAMES:
            return previous_zone, round(dwell_time, 1), None

        # ── stable zone change ────────────────────────────────────────────
        if previous_zone != "Checkout":
            ds = self.cumulative_dwell_by_zone.setdefault(track_id, {})
            ds[previous_zone] = ds.get(previous_zone, 0.0) + max(0.0, dwell_time)

        transition_event = self._ev({
            "event_type": "ZONE_TRANSITION",
            "track_id":   customer_id,
            "from_zone":  previous_zone,
            "to_zone":    current_zone,
            "dwell_sec":  round(dwell_time, 1),
        })

        self.active_tracks[track_id] = {
            "zone": current_zone, "entry_time": current_time
        }
        self.pending_zones.pop(track_id, None)
        self.pending_counts.pop(track_id, None)
        self.missing_counts[track_id] = 0

        self.zone_history.setdefault(track_id, [])
        if (not self.zone_history[track_id]
                or self.zone_history[track_id][-1] != current_zone):
            self.zone_history[track_id].append(current_zone)

        self.triggered_events.pop(track_id, None)
        return current_zone, 0.0, transition_event

    # ── long dwell alert ──────────────────────────────────────────────────

    def check_events(self, track_id, zone_name, dwell_sec):
        if zone_name == "Shelf A" and dwell_sec > 30:
            if not self.triggered_events.get(track_id):
                self.triggered_events[track_id] = True
                return self._ev({
                    "event_type": "LONG_DWELL",
                    "track_id":   self.get_customer_id(track_id),
                    "zone_name":  zone_name,
                    "dwell_sec":  round(dwell_sec, 1),
                })
        return None

    # ── track reset ───────────────────────────────────────────────────────

    def reset_track(self, track_id):
        for d in (self.active_tracks, self.triggered_events, self.missing_counts,
                  self.pending_zones, self.pending_counts, self.zone_history,
                  self.cumulative_dwell_by_zone, self.active_customer_ids):
            d.pop(track_id, None)

    # ── cleanup missing ───────────────────────────────────────────────────

    def cleanup_missing_tracks(self, current_track_ids):
        missing_tracks = []
        missing_events = []
        current_track_ids = set(current_track_ids)
        MAX_MISSED_FRAMES = 15

        for track_id in list(self.active_tracks.keys()):
            if track_id in current_track_ids:
                self.missing_counts[track_id] = 0
                continue

            self.missing_counts[track_id] = (
                self.missing_counts.get(track_id, 0) + 1
            )
            if self.missing_counts[track_id] < MAX_MISSED_FRAMES:
                continue

            # Real exit
            track_data  = self.active_tracks.pop(track_id)
            zone        = track_data["zone"]
            dwell_time  = time.time() - track_data["entry_time"]
            customer_id = self.get_customer_id(track_id)
            if customer_id is None:
                customer_id = self._create_customer_id(track_id)

            missing_tracks.append(customer_id)
            self.completed_tracks.add(track_id)

            if zone is not None:
                if zone != "Checkout":
                    ds = self.cumulative_dwell_by_zone.setdefault(track_id, {})
                    ds[zone] = ds.get(zone, 0.0) + max(0.0, dwell_time)

                missing_events.append(self._ev({
                    "event_type": "ZONE_EXIT",
                    "track_id":   customer_id,
                    "zone_name":  zone,
                    "dwell_sec":  round(dwell_time, 1),
                }))

            for d in (self.triggered_events, self.missing_counts,
                      self.pending_zones, self.pending_counts):
                d.pop(track_id, None)

        return missing_tracks, missing_events

    # ── zone history ──────────────────────────────────────────────────────

    def track_zone_transition(self, track_id, current_zone):
        if current_zone is None:
            return None
        if track_id not in self.zone_history:
            self.zone_history[track_id] = [current_zone]
            return self._ev({
                "event_type": "ZONE_ENTRY",
                "track_id":   self.get_customer_id(track_id),
                "zone_name":  current_zone,
            })
        previous_zone = self.zone_history[track_id][-1]
        if previous_zone == current_zone:
            return None
        self.zone_history[track_id].append(current_zone)
        return self._ev({
            "event_type": "ZONE_TRANSITION",
            "track_id":   self.get_customer_id(track_id),
            "from_zone":  previous_zone,
            "to_zone":    current_zone,
        })

    # ── live ML features ──────────────────────────────────────────────────

    def get_live_customer_features(self, track_id):
        if track_id not in self.active_tracks:
            return None

        customer_id    = self._get_or_create_customer_id(track_id)
        dwell_by_zone  = self.cumulative_dwell_by_zone.get(track_id, {})
        active         = self.active_tracks[track_id]
        current_zone   = active.get("zone")
        live_dwell     = max(0.0, time.time() - active.get("entry_time", time.time()))

        shelf_a_dwell  = dwell_by_zone.get("Shelf A",  0.0)
        entrance_dwell = dwell_by_zone.get("Entrance", 0.0)
        if current_zone == "Shelf A":
            shelf_a_dwell  += live_dwell
        elif current_zone == "Entrance":
            entrance_dwell += live_dwell

        history           = self.zone_history.get(track_id, [])
        non_checkout      = [z for z in history if z and z != "Checkout"]
        zones_visited     = len(set(non_checkout))
        transitions       = max(0, len(history) - 1)
        total_dwell       = shelf_a_dwell + entrance_dwell

        return {
            "track_id":      customer_id,
            "total_dwell":   round(total_dwell, 1),
            "shelf_a_dwell": round(shelf_a_dwell, 1),
            "entrance_dwell": round(entrance_dwell, 1),
            "zones_visited": zones_visited,
            "transitions":   transitions,
        }
