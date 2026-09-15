from sklearn.ensemble import IsolationForest

from src.ml.feature_engineering_v2 import FeatureEngineerV2


class AnomalyDetector:
    def __init__(self, event_store, contamination=0.15):
        self.event_store = event_store
        self.engineer = FeatureEngineerV2(event_store)
        self.contamination = contamination

    def _rows(self):
        rows = []
        ids = []
        for track_id in self.event_store.get_customer_ids():
            f = self.engineer.get_customer_features(track_id)
            if (
                f["total_dwell"] == 0
                and f["zones_visited"] == 0
                and f["transitions"] == 0
            ):
                continue
            rows.append([
                f["total_dwell"],
                f["shelf_a_dwell"],
                f["entrance_dwell"],
                f["zones_visited"],
                f["transitions"],
            ])
            ids.append(track_id)
        return ids, rows

    def detect(self):
        ids, rows = self._rows()
        if not rows:
            return []

        if len(rows) < 5:
            return [
                {"track_id": track_id, "status": "INSUFFICIENT_DATA", "score": 0.0}
                for track_id in ids
            ]

        model = IsolationForest(
            contamination=min(self.contamination, max(1 / len(rows), 0.01)),
            random_state=42,
        )
        labels = model.fit_predict(rows)
        scores = model.decision_function(rows)

        return [
            {
                "track_id": track_id,
                "status": "ANOMALY" if label == -1 else "NORMAL",
                "score": round(float(score), 4),
            }
            for track_id, label, score in zip(ids, labels, scores)
        ]
