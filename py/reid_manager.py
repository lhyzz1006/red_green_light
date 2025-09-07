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

        best_id, best_score = None, None
        for tid, feat in self.gallery_features.items():
            score = float(np.dot(new_feat, feat))
            print(f"当前与 ID={tid} 的余弦相似度: {score:.4f}")
            if score > float(self.threshold) and (best_score is None or score > best_score):
                print(f"匹配更新！原 best_score={best_score}，新为 {score:.4f}，ID={tid}")
                best_id = tid
                best_score = score
            else:
                print(f"未更新：score={score:.4f} ≤ threshold({self.threshold}) 或 ≤ best_score={best_score}")

        print(f"[匹配结果] 当前匹配结果为：ID={best_id}, 匹配得分={best_score}")
        return best_id, best_score if best_score is not None else -1
