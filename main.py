import json
import joblib
from src.prepare_data import crop_and_convert_vnts, group_split_data
from src.features import extract_hog_features
from src.train_eval import train_and_eval_all, plot_and_save_cm

def main():
    print("=== BƯỚC 1: KHẢO SÁT & CROP DATASET VNTS ===")
    meta = crop_and_convert_vnts()
    if len(meta) == 0:
        print("LỖI: Chưa có ảnh trong data/vnts_raw/. Hãy chép ảnh vào!")
        return

    train_meta, val_meta, test_meta = group_split_data(meta)

    print("\n=== BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG HOG ===")
    X_train = extract_hog_features([m["crop_path"] for m in train_meta])
    y_train = [m["label"] for m in train_meta]
    
    X_val = extract_hog_features([m["crop_path"] for m in val_meta])
    y_val = [m["label"] for m in val_meta]
    
    X_test = extract_hog_features([m["crop_path"] for m in test_meta])
    y_test = [m["label"] for m in test_meta]

    print("\n=== BƯỚC 3: HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH ===")
    results = train_and_eval_all(X_train, y_train, X_val, y_val)

    # Chọn mô hình tốt nhất theo Macro F1 trên tập Val
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = results[best_name]["model"]
    print(f"\n=> Mô hình tốt nhất trên tập Validation: {best_name}")

    print("\n=== BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TEST SET ===")
    y_test_pred = best_model.predict(X_test)
    labels = sorted(list(set(y_train)))
    plot_and_save_cm(y_test, y_test_pred, labels)

    print("\n=== BƯỚC 5: LƯU MODEL VÀ CONFIG ===")
    joblib.dump(best_model, "models/best_model.joblib")
    config = {
        "input_width": 48,
        "input_height": 48,
        "color": "grayscale",
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2]
    }
    with open("models/config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Đã hoàn tất lưu vết pipeline!")

if __name__ == "__main__":
    main()