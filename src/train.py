import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu cua bo du lieu Adult va do lech cho phep
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def check_label_drift(y_train: pd.Series) -> float:
    """Tinh ty le lop duong va canh bao neu lech qua DRIFT_TOLERANCE so voi tham chieu."""
    positive_rate = float(y_train.mean())
    drift = abs(positive_rate - REFERENCE_POSITIVE_RATE)
    if drift > DRIFT_TOLERANCE:
        print(
            f"CANH BAO LECH DU LIEU: ty le lop duong = {positive_rate:.1%}, "
            f"lech {drift * 100:.1f} diem phan tram so voi tham chieu "
            f"{REFERENCE_POSITIVE_RATE:.1%} (nguong {DRIFT_TOLERANCE * 100:.0f} diem)."
        )
    else:
        print(f"Ty le lop duong: {positive_rate:.1%} (trong nguong cho phep).")
    return positive_rate


def find_best_threshold(y_eval: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    """Bonus 2: quet nguong quyet dinh tu 0.10 den 0.90 (buoc 0.05), tra ve (nguong, f1) tot nhat."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2):
        score = f1_score(y_eval, (proba >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """
    # Mac dinh ghi vao SQLite cuc bo; dat MLFLOW_TRACKING_URI de dung server tu xa (Bonus 1).
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_rate = check_label_drift(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(df_train))

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        best_threshold, best_f1 = find_best_threshold(y_eval, model.predict_proba(X_eval)[:, 1])

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_best_threshold", best_f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Nguong toi uu: {best_threshold:.2f} -> F1 {best_f1:.4f} (nguong 0.5: F1 {f1:.4f})")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "positive_rate": positive_rate,
                    "best_threshold": best_threshold,
                    "f1_best_threshold": best_f1,
                    "train_rows": len(df_train),
                },
                f,
                indent=2,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
