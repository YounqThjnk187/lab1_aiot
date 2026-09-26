import os
import cv2
import glob
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    xc_px, yc_px = xc * img_w, yc * img_h
    w_px, h_px = bw * img_w, bh * img_h
    x1 = int(max(0, xc_px - w_px / 2))
    y1 = int(max(0, yc_px - h_px / 2))
    x2 = int(min(img_w, xc_px + w_px / 2))
    y2 = int(min(img_h, yc_px + h_px / 2))
    return x1, y1, x2, y2

def crop_and_convert_vnts(raw_dir="data/vnts_raw", output_dir="data/vnts_classification", min_size=16, margin=0.05):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(raw_path.rglob("*.jpg")) + list(raw_path.rglob("*.png"))
    crop_count = 0
    ignored_count = 0
    metadata = []

    for img_p in image_paths:
        txt_p = img_p.with_suffix(".txt")
        if not txt_p.exists():
            continue
        
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w, _ = img.shape
        
        with open(txt_p, "r") as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            xc, yc, bw, bh = map(float, parts[1:5])
            
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            
            crop_w, crop_h = x2 - x1, y2 - y1
            x1_m = max(0, int(x1 - margin * crop_w))
            y1_m = max(0, int(y1 - margin * crop_h))
            x2_m = min(w, int(x2 + margin * crop_w))
            y2_m = min(h, int(y2 + margin * crop_h))
            
            roi = img[y1_m:y2_m, x1_m:x2_m]
            
            if roi.shape[0] < min_size or roi.shape[1] < min_size:
                ignored_count += 1
                continue
                
            cls_folder = out_path / cls_id
            cls_folder.mkdir(exist_ok=True)
            
            crop_name = f"{img_p.stem}_crop_{idx}.jpg"
            save_p = cls_folder / crop_name
            cv2.imwrite(str(save_p), roi)
            
            metadata.append({
                "crop_path": str(save_p),
                "label": int(cls_id),
                "source_image": img_p.name
            })
            crop_count += 1
            
    print(f"[Prepare] Tạo {crop_count} ảnh crop. Bỏ qua {ignored_count} crop quá nhỏ (<{min_size}px).")
    return metadata

def group_split_data(metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    groups = [m["source_image"] for m in metadata]
    labels = [m["label"] for m in metadata]
    
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(metadata, labels, groups))
    
    train_val_meta = [metadata[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]
    
    val_adj_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_adj_ratio, random_state=seed)
    train_idx, val_idx = next(gss_val.split(train_val_meta, train_val_labels, train_val_groups))
    
    train_data = [train_val_meta[i] for i in train_idx]
    val_data = [train_val_meta[i] for i in val_idx]
    test_data = [metadata[i] for i in test_idx]
    
    print(f"[Split] Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data