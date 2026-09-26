import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000))
        ])
    }

def train_and_eval_all(X_train, y_train, X_val, y_val):
    models = get_models()
    results = {}
    
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_val)
        infer_time = (time.perf_counter() - t0) / len(X_val) * 1000  # ms/sample
        
        acc = accuracy_score(y_val, y_pred)
        b_acc = balanced_accuracy_score(y_val, y_pred)
        m_f1 = f1_score(y_val, y_pred, average="macro")
        
        results[name] = {
            "model": model,
            "acc": acc,
            "b_acc": b_acc,
            "macro_f1": m_f1,
            "train_time": train_time,
            "infer_time_ms": infer_time
        }
        print(f"[{name}] Acc: {acc:.4f} | Macro F1: {m_f1:.4f} | Train Time: {train_time:.2f}s | Infer: {infer_time:.3f}ms/sample")
        
    return results

def plot_and_save_cm(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    # 1. Tìm danh sách các nhãn thực sự xuất hiện trong dữ liệu test
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # 2. Lấy danh sách tên class tương ứng
    display_labels = [labels[i] for i in unique_labels] if isinstance(labels, list) else unique_labels
    
    # 3. Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # 4. Vẽ confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    # Bỏ values_format=".2f" nếu cm là số nguyên, hoặc giữ nguyên nếu là chuẩn hóa
    disp.plot(ax=ax, cmap="Blues", xticks_rotation='vertical') 
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()