class FeatureEngineerV2:
    """Build leakage-aware pre-checkout customer features."""

    FEATURES = [
        "total_dwell",
        "shelf_a_dwell",
        "entrance_dwell",
        "zones_visited",
        "transitions",
    ]

    def __init__(self, event_store):
        self.event_store = event_store

    def get_customer_features(self, track_id):
        events = self.event_store.get_customer_events(track_id)

        entrance = 0.0
        shelf_a = 0.0
        zones = []
        transitions = 0
        reached_checkout = False

        for event in events:
            event_type = event[2]
            from_zone = event[3]
            to_zone = event[4]
            zone_name = event[5]
            dwell = float(event[6] or 0.0)

            # Stop feature extraction at first checkout.
            if (
                (event_type == "ZONE_ENTRY" and zone_name == "Checkout")
                or
                (event_type == "ZONE_TRANSITION" and to_zone == "Checkout")
            ):
                reached_checkout = True
                break

            if event_type == "ZONE_ENTRY":
                if zone_name and zone_name not in zones:
                    zones.append(zone_name)

            elif event_type == "ZONE_TRANSITION":
                transitions += 1
                if to_zone and to_zone not in zones:
                    zones.append(to_zone)

                # Transition dwell belongs to the zone being left.
                if from_zone == "Entrance":
                    entrance += dwell
                elif from_zone == "Shelf A":
                    shelf_a += dwell

            elif event_type == "ZONE_EXIT":
                if zone_name == "Entrance":
                    entrance += dwell
                elif zone_name == "Shelf A":
                    shelf_a += dwell

            # LONG_DWELL is an alert, not additional dwell.

        total = entrance + shelf_a

        return {
            "total_dwell": round(total, 2),
            "shelf_a_dwell": round(shelf_a, 2),
            "entrance_dwell": round(entrance, 2),
            "zones_visited": len(set(zones)),
            "transitions": transitions,
            "reached_checkout": int(reached_checkout),
        }

    def prepare_ml_data(self):
        X, y = [], []

        for track_id in self.event_store.get_customer_ids():
            events = self.event_store.get_customer_events(track_id)
            if not events:
                continue

            features = self.get_customer_features(track_id)

            # Direct checkout gives no useful pre-checkout behavior.
            first_meaningful = next(
                (
                    e for e in events
                    if e[2] in {"ZONE_ENTRY", "ZONE_TRANSITION", "ZONE_EXIT"}
                ),
                None,
            )
            if first_meaningful is None:
                continue

            has_precheckout_behavior = (
                features["total_dwell"] > 0
                or features["zones_visited"] > 0
                or features["transitions"] > 0
            )
            if not has_precheckout_behavior:
                continue

            X.append([
                features[name]
                for name in self.FEATURES
            ])
            y.append(features["reached_checkout"])

        return X, y
