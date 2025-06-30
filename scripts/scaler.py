import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_all_segments(base_dir, target_len=300):
    X = []
    for category in ["cheater", "legit"]:
        path = os.path.join(base_dir, category)
        for file in os.listdir(path):
            if not file.endswith('.csv'):
                continue
            try:
                df = pd.read_csv(os.path.join(path, file))
                drop_cols = ["tick", "steamid", "label", "weapon_name", "weapon_type"]
                df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

                if df.shape[0] < target_len:
                    pad_rows = target_len - df.shape[0]
                    last_row = df.iloc[[-1]].copy()
                    padding = pd.concat([last_row] * pad_rows, ignore_index=True)
                    df = pd.concat([df, padding], ignore_index=True)

                X.append(df.values[:target_len])
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return np.array(X)

if __name__ == "__main__":
    base_dir = "data/processed/features"
    print("Loading segments...")
    X = load_all_segments(base_dir)

    print("Fitting scaler...")
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    scaler.fit(X_flat)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")
