import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image_paths, img_size=(48, 48), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    features = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        hog_feat = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        )
        features.append(hog_feat)
    return np.array(features)