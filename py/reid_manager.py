import torch
import numpy as np
from torchvision import models, transforms
import torch.nn.functional as F
import cv2

class ReIDManager:
    def __init__(self, threshold=0.6):
        self.gallery_features = {}  # track_id -> 特征向量
        self.threshold = threshold

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, image_np):
        if image_np is None or image_np.shape[0] == 0 or image_np.shape[1] == 0:
            return None
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image = self.transform(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(image)
            feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy()

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
