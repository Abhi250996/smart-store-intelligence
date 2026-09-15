"""
StoreAnalytics — aggregated KPIs for Smart Store Intelligence
=============================================================

All original methods preserved.
New *_for() methods accept date_str / camera_id for historical queries.
"""

from __future__ import annotations

from typing import Optional


class StoreAnalytics:
    def __init__(self, event_store):
        self.event_store = event_store

    # ── original methods (unchanged) ─────────────────────────────────────

    def get_total_customer(self) -> int:
        return self.event_store.get_total_customer()

    def get_most_visited_zone(self) -> dict:
        counts = self.event_store.get_customers_most_visit_zone()
        if not counts:
            return {"zone": "N/A", "count": 0}
        zone, count = max(counts.items(), key=lambda x: x[1])
        return {"zone": zone, "count": count}

    def get_average_dwell_time(self) -> dict:
        return self.event_store.get_average_dwell_by_zone()

    def get_checkout_customers(self) -> int:
        return self.event_store.get_checkout_customers()

    def get_zone_engagement_rate(self) -> dict:
        counts = self.event_store.get_customers_most_visit_zone()
        total  = self.get_total_customer()
        if total <= 0:
            return {z: 0.0 for z in counts}
        return {z: round(c / total * 100, 2) for z, c in counts.items()}

    def get_zone_transition(self) -> dict:
        return self.event_store.get_zone_transitions()

    # ── date / camera aware methods ───────────────────────────────────────

    def get_total_customer_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> int:
        return self.event_store.get_total_customer_by_date(date_str, camera_id)

    def get_checkout_customers_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> int:
        return self.event_store.get_checkout_customers_by_date(date_str, camera_id)

    def get_most_visited_zone_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        counts = self.event_store.get_customers_most_visit_zone_by_date(
            date_str, camera_id
        )
        if not counts:
            return {"zone": "N/A", "count": 0}
        zone, count = max(counts.items(), key=lambda x: x[1])
        return {"zone": zone, "count": count}

    def get_zone_engagement_rate_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        counts = self.event_store.get_customers_most_visit_zone_by_date(
            date_str, camera_id
        )
        total = self.get_total_customer_for(date_str, camera_id)
        if total <= 0:
            return {z: 0.0 for z in counts}
        return {z: round(c / total * 100, 2) for z, c in counts.items()}

    def get_average_dwell_time_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        return self.event_store.get_average_dwell_by_zone_by_date(
            date_str, camera_id
        )

    def get_zone_transitions_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> dict:
        return self.event_store.get_zone_transitions_by_date(date_str, camera_id)

    def get_peak_hour_for(
        self,
        date_str: Optional[str],
        camera_id: Optional[str] = None,
    ) -> Optional[str]:
        return self.event_store.get_peak_hour_by_date(date_str, camera_id)
