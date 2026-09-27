import os
import time
import joblib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ==========================================
# 0. CẤU HÌNH ĐƯỜNG DẪN & CONTRACT
# ==========================================
BASE_DIR = Path(__file__).parent
DATA_CROP_DIR = BASE_DIR / "data" / "vnts_classification"  # Đã sửa thành vnts_classification
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PREPROCESSING_CONTRACT = {
    'target_size': (48, 48),  # Đã đồng bộ về 48x48 theo main.py
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}

stage_times = {}

print("=== BẮT ĐẦU BENCHMARK THỜI GIAN TRÊN LOCAL PC ===")

# ==========================================
# STAGE 1: ĐỌC DỮ LIỆU CROP & TIỀN XỬ LÝ (IO & RESIZE)
# ==========================================
t0 = time.time()

# Lấy danh sách ảnh crop trong thư mục
image_paths = list(DATA_CROP_DIR.rglob("*.jpg")) + list(DATA_CROP_DIR.rglob("*.png"))
if len(image_paths) == 0:
    print(f"LỖI: Không tìm thấy ảnh trong {DATA_CROP_DIR}. Vui lòng kiểm tra lại dữ liệu!")
    exit()

images_resized = []
labels = []

# Giả định cấu trúc thư mục dạng data/vnts_classification/class_id/image.jpg
for img_path in image_paths:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized = cv2.resize(img, PREPROCESSING_CONTRACT['target_size'], interpolation=cv2.INTER_AREA)
        images_resized.append(resized)
        # Lấy nhãn từ tên thư mục chứa ảnh
        label = img_path.parent.name
        labels.append(label)

t1 = time.time()
stage_times['Stage 1: Preprocessing & Load'] = round(t1 - t0, 4)
print(f"✓ Done Stage 1: Đã đọc {len(images_resized)} ảnh ({stage_times['Stage 1: Preprocessing & Load']}s)")

# ==========================================
# STAGE 2: TRÍCH XUẤT ĐẶC TRƯNG HOG
# ==========================================
t0 = time.time()

X_features = []
for img in images_resized:
    feat = hog(
        img,
        orientations=PREPROCESSING_CONTRACT['orientations'],
        pixels_per_cell=PREPROCESSING_CONTRACT['pixels_per_cell'],
        cells_per_block=PREPROCESSING_CONTRACT['cells_per_block'],
        block_norm=PREPROCESSING_CONTRACT['block_norm'],
        transform_sqrt=PREPROCESSING_CONTRACT['transform_sqrt'],
        feature_vector=True
    )
    X_features.append(feat)

X = np.array(X_features)
y = np.array(labels)

t1 = time.time()
stage_times['Stage 2: Feature Extraction (HOG)'] = round(t1 - t0, 4)
print(f"✓ Done Stage 2: Trích xuất HOG xong ({stage_times['Stage 2: Feature Extraction (HOG)']}s)")

# Chia dữ liệu giả định 80% Train / 20% Test để benchmark
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==========================================
# STAGE 3: HUẤN LUYỆN MÔ HÌNH (LINEAR SVM)
# ==========================================
t0 = time.time()

model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
model.fit(X_train, y_train)

t1 = time.time()
stage_times['Stage 3: Model Training (Linear SVM)'] = round(t1 - t0, 4)
print(f"✓ Done Stage 3: Huấn luyện xong ({stage_times['Stage 3: Model Training (Linear SVM)']}s)")

# ==========================================
# STAGE 4: INFERENCE TRÊN TẬP TEST (BATCH)
# ==========================================
t0 = time.time()

y_pred = model.predict(X_test)

t1 = time.time()
stage_times['Stage 4: Test Inference (Batch)'] = round(t1 - t0, 4)
acc = accuracy_score(y_test, y_pred)
print(f"✓ Done Stage 4: Dự đoán {len(X_test)} mẫu xong (Acc: {acc*100:.2f}%) ({stage_times['Stage 4: Test Inference (Batch)']}s)")

# ==========================================
# STAGE 5: LƯU ARTIFACT (MODEL + CONTRACT)
# ==========================================
t0 = time.time()

artifact = {
    'model': model,
    'contract': PREPROCESSING_CONTRACT
}
joblib.dump(artifact, OUTPUT_DIR / "vnts_pipeline_svm.pkl")

t1 = time.time()
stage_times['Stage 5: Save Artifact'] = round(t1 - t0, 4)
print(f"✓ Done Stage 5: Lưu file pkl xong ({stage_times['Stage 5: Save Artifact']}s)")

# ==========================================
# BÁO CÁO TỔNG HỢP THỜI GIAN
# ==========================================
df_result = pd.DataFrame(list(stage_times.items()), columns=['Giai đoạn (Stage)', 'Thời gian (s)'])
total_time = sum(stage_times.values())

print("\n" + "="*50)
print("BẢNG KẾT QUẢ THỜI GIAN THỰC THI TRÊN LOCAL PC")
print("="*50)
print(df_result.to_string(index=False))
print("-" * 50)
print(f"TỔNG THỜI GIAN: {total_time:.4f} giây")
print("="*50)

# Lưu kết quả đo ra file CSV
df_result.to_csv(OUTPUT_DIR / "local_benchmark_results.csv", index=False)