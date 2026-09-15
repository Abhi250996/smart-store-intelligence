from src.event_store import EventStore
from src.ml.feature_engineering_v2 import FeatureEngineerV2


def main():
    store = EventStore()
    engineer = FeatureEngineerV2(store)
    X, y = engineer.prepare_ml_data()

    total_customers = len(store.get_customer_ids())
    usable = len(X)
    positives = sum(y)
    negatives = usable - positives
    direct_checkout = max(total_customers - usable - 0, 0)

    print("=" * 60)
    print("SMART STORE DATASET CHECK")
    print("=" * 60)
    print("Total customers       :", total_customers)
    print("Usable customers      :", usable)
    print("Checkout (target=1)   :", positives)
    print("No-checkout (target=0):", negatives)
    if usable:
        print("Checkout class        :", round(positives / usable * 100, 1), "%")
        print("STATUS                :", "READY FOR FIRST EXPERIMENT")
    else:
        print("STATUS                :", "NOT READY")


if __name__ == "__main__":
    main()
