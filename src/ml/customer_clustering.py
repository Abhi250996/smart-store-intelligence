from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.ml.feature_engineering_v2 import FeatureEngineerV2


class CustomerClustering:
    def __init__(self, event_store):
        self.event_store = event_store
        self.engineer = FeatureEngineerV2(event_store)

    def cluster(self, n_clusters=3):
        ids = []
        rows = []

        for track_id in self.event_store.get_customer_ids():
            f = self.engineer.get_customer_features(track_id)
            if (
                f["total_dwell"] == 0
                and f["zones_visited"] == 0
                and f["transitions"] == 0
            ):
                continue

            ids.append(track_id)
            rows.append([
                f["total_dwell"],
                f["shelf_a_dwell"],
                f["entrance_dwell"],
                f["zones_visited"],
                f["transitions"],
            ])

        if len(rows) < 3:
            return []

        n_clusters = min(n_clusters, len(rows))
        scaled = StandardScaler().fit_transform(rows)
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(scaled)

        return [
            {"track_id": track_id, "cluster": int(label)}
            for track_id, label in zip(ids, labels)
        ]

    def detect(self, n_clusters=3):
        return self.cluster(n_clusters=n_clusters)
