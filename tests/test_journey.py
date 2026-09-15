from src.customer_journey import CustomerJourneyAnalyzer


class FakeStore:
    def get_customer_events(self, track_id):
        return [
            (1, track_id, "ZONE_ENTRY", None, None, "Entrance", 0.0, ""),
            (2, track_id, "ZONE_TRANSITION", "Entrance", "Shelf A", None, 2.0, ""),
            (3, track_id, "ZONE_TRANSITION", "Shelf A", "Checkout", None, 4.0, ""),
        ]


def test_customer_path():
    analyzer = CustomerJourneyAnalyzer(FakeStore())
    assert analyzer.get_customer_path(10) == ["Entrance", "Shelf A", "Checkout"]
    assert analyzer.get_zones_visited(10) == ["Entrance", "Shelf A", "Checkout"]
