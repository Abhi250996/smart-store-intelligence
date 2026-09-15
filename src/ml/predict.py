"""
ConversionPredictor
===================
Loads the sklearn Pipeline model ONCE.
The model is a joblib-pickled sklearn Pipeline (StandardScaler + LogisticRegression).

Design rules:
- joblib.load() is called exactly ONCE in __init__.
- __init__ is called ONCE per session from dashboard._init_session_state().
- predict_live_features() and predict_customer() do NOT reload the model.
- An optional shared EventStore can be injected to avoid creating extra
  SQLite connections; if None a private one is created.
"""

import joblib

from src.event_store import EventStore
from src.ml.feature_engineering_v2 import FeatureEngineerV2

_MODEL_PATH = "models/customer_conversion_model.pkl"


class ConversionPredictor:

    def __init__(self, event_store: EventStore | None = None):
        # This is the ONLY place joblib.load is called in the entire project.
        # The InconsistentVersionWarning will print once here at session start.
        self.model = joblib.load(_MODEL_PATH)

        # Reuse a shared EventStore if provided; create a private one otherwise.
        self.event_store = event_store if event_store is not None else EventStore()
        self.feature_engineer = FeatureEngineerV2(self.event_store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_vector(customer: dict) -> list:
        return [[
            customer["total_dwell"],
            customer["shelf_a_dwell"],
            customer["entrance_dwell"],
            customer["zones_visited"],
            customer["transitions"],
        ]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_live_features(self, customer: dict) -> dict | None:
        """Predict from live features supplied directly (e.g. from EventEngine)."""
        if customer is None:
            return None
        features = self._feature_vector(customer)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]
        return {
            "track_id": customer["track_id"],
            "prediction": int(prediction),
            "conversion_probability": round(probability * 100, 2),
        }

    def predict_customer(self, track_id) -> dict | None:
        """Predict from stored historical events for a given track_id."""
        customer = self.feature_engineer.get_customer_features(track_id)
        return self.predict_live_features({
            "track_id": track_id,
            "total_dwell": customer["total_dwell"],
            "shelf_a_dwell": customer["shelf_a_dwell"],
            "entrance_dwell": customer["entrance_dwell"],
            "zones_visited": customer["zones_visited"],
            "transitions": customer["transitions"],
        })


# ------------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    predictor = ConversionPredictor()
    customer_ids = predictor.event_store.get_customer_ids()

    if customer_ids:
        print("\nCustomer Predictions")
        print("====================")
        for track_id in customer_ids:
            result = predictor.predict_customer(track_id)
            print(f"Customer {result['track_id']} -> {result['conversion_probability']}%")
    else:
        print("No customers found.")
