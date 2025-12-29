import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.preprocessing import preprocess
from filelock import FileLock

DATA_PATH = "data/incoming_data.csv"
LOCK_PATH = "data/incoming_data.csv.lock"

def retrain():
    # 1. READ WITH LOCK
    try:
        lock = FileLock(LOCK_PATH)
        with lock.acquire(timeout=10):
            data = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Lock error: {e}")
        return None

    # 2. Filter valid data
    data = data.dropna(subset=["label"])
    
    # Check 1: Enough rows?
    if len(data) < 50:  # Increased limit for the bigger IoT dataset
        return None

    # Check 2: Do we have multiple classes?
    if data["label"].nunique() < 2:
        return {
            "status": "skipped",
            "reason": "Data needs multiple classes to train."
        }

    # 3. DYNAMIC FEATURES: Select everything that isn't 'label'
    feature_cols = [c for c in data.columns if c != "label"]
    
    # Preprocess
    X = preprocess(data[feature_cols], training=True)
    y = data["label"]

    # 4. Split & Train
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        eval_metric="mlogloss" # Better for multiclass
    )
    
    model.fit(X_train, y_train)

    # 5. Metrics & Save
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    joblib.dump(model, "model/current_model.pkl")

    return {
        "status": "replaced",
        "accuracy": f"{acc:.2%}",
        "samples_used": len(data)
    }