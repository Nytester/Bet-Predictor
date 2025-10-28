import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump
from utils.feature_engineering import build_feature_frame, extract_target

DATA_PATH = os.path.join("data", "sample_past.csv")   # replace with your file e.g., past_games.csv
OUT_PATH  = os.path.join("model", "predictor.pkl")

def main():
    print(f"[train] Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = build_feature_frame(df, training=True)
    y = extract_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"[train] ROC AUC: {auc:.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    dump(clf, OUT_PATH)
    print(f"[train] Saved model to: {OUT_PATH}")

if __name__ == "__main__":
    main()
