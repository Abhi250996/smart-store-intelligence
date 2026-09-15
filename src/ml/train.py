import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.event_store import EventStore
from src.ml.feature_engineering_v2 import FeatureEngineerV2


def train_model():
    print("=" * 70)
    print("SMART STORE - CUSTOMER CHECKOUT PREDICTION")
    print("=" * 70)

    store = EventStore()
    engineer = FeatureEngineerV2(store)
    X, y = engineer.prepare_ml_data()

    if len(X) < 9 or len(set(y)) < 2:
        raise RuntimeError(
            "Not enough usable data. Collect more customer journeys first."
        )

    print("\nTotal samples:", len(X))
    print("Checkout samples:", sum(y))
    print("No-checkout samples:", len(y) - sum(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    path = "models/customer_conversion_model.pkl"
    joblib.dump(model, path)
    print(f"\nModel saved: {path}")


if __name__ == "__main__":
    train_model()
