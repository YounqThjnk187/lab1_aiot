import cv2
import json
import joblib
import numpy as np
from skimage.feature import hog

def run_inference(image_path, model_path="models/best_model.joblib", config_path="models/config.json"):
    model = joblib.load(model_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Không đọc được ảnh!")
    img_resized = cv2.resize(img, (cfg["input_width"], cfg["input_height"]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    feat = hog(
        gray,
        orientations=cfg["orientations"],
        pixels_per_cell=cfg["pixels_per_cell"],
        cells_per_block=cfg["cells_per_block"],
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    ).reshape(1, -1)
    
    pred_class_idx = model.predict(feat)[0]
    return pred_class_idx