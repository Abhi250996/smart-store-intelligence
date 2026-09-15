class CustomerJourneyAnalyzer:
    def __init__(self, event_store):
        self.event_store = event_store

    def get_customer_events(self, track_id):
        return self.event_store.get_customer_events(track_id)

    def get_zones_visited(self, track_id):
        events = self.get_customer_events(track_id)
        zones = []
        for event in events:
            event_type = event[2]
            if event_type == "ZONE_ENTRY":
                zone = event[5]
                if zone and zone not in zones:
                    zones.append(zone)
            elif event_type == "ZONE_TRANSITION":
                zone = event[4]
                if zone and zone not in zones:
                    zones.append(zone)
        return zones

    def get_customer_path(self, track_id):
        events = self.get_customer_events(track_id)
        path = []
        for event in events:
            event_type = event[2]
            if event_type == "ZONE_ENTRY":
                zone = event[5]
            elif event_type == "ZONE_TRANSITION":
                zone = event[4]
            else:
                continue
            if zone and (not path or path[-1] != zone):
                path.append(zone)
        return path

    def get_total_dwell(self, track_id):
        events = self.get_customer_events(track_id)
        total = 0.0
        for event in events:
            if event[2] == "ZONE_EXIT":
                total += float(event[6] or 0.0)
            elif event[2] == "ZONE_TRANSITION":
                total += float(event[6] or 0.0)
        return round(total, 2)

    def get_customer_summary(self, track_id):
        return {
            "track_id": track_id,
            "customer_events": self.get_customer_events(track_id),
            "zones_visited": self.get_zones_visited(track_id),
            "customer_path": self.get_customer_path(track_id),
            "total_dwell": self.get_total_dwell(track_id),
        }
