"""
Bonus 3: bao cao precision / recall chi tiet cho tung lop.

Doc models/model.joblib (do src/train.py tao ra), danh gia tren tap holdout va
ghi confusion matrix + precision / recall tung lop vao outputs/detail.txt.
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["thu_nhap_thap", "thu_nhap_cao"]


def evaluate(
    model_path: str = "models/model.joblib",
    eval_path: str = "data/holdout.csv",
    out_path: str = "outputs/detail.txt",
) -> str:
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    preds = model.predict(X_eval)

    (tn, fp), (fn, tp) = confusion_matrix(y_eval, preds, labels=[0, 1])
    text = "\n".join(
        [
            "CONFUSION MATRIX (hang = thuc te, cot = du doan)",
            f"{'':>16}{'pred_thap':>12}{'pred_cao':>12}",
            f"{'thuc_te_thap':>16}{tn:>12}{fp:>12}",
            f"{'thuc_te_cao':>16}{fn:>12}{tp:>12}",
            "",
            f"Bo sot nguoi thu nhap cao (FN): {fn}",
            f"Gan nham nguoi thu nhap thap (FP): {fp}",
            "",
            "PRECISION / RECALL TUNG LOP",
            classification_report(y_eval, preds, labels=[0, 1], target_names=LABELS, digits=4),
        ]
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(evaluate(*sys.argv[1:]))
