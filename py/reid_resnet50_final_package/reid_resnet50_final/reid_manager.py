import numpy as np
from feature_extractor import FeatureExtractor

class ReIDManager:
    def __init__(self, threshold=0.5):
        self.gallery_features = {}  # track_id -> 特征向量
        self.threshold = threshold
        self.extractor = FeatureExtractor()

    def extract_feature(self, image_np):
        return self.extractor(image_np)

    def update_feature(self, track_id, image_np):
        feat = self.extract_feature(image_np)
        if feat is not None:
            self.gallery_features[int(track_id)] = feat

    def match_feature(self, image_np):
        new_feat = self.extract_feature(image_np)
        if new_feat is None:
            return None, -1
        best_id, best_score = None, -1
        for tid, feat in self.gallery_features.items():
            score = float(np.dot(new_feat, feat))
            print(f"[🔎 对比] 当前与 ID={tid} 的余弦相似度: {score:.4f}")
            if score > self.threshold and score > best_score:
                best_id, best_score = tid, score
        return best_id, best_score