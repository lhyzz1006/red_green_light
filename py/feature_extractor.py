import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 使用 torch.hub 加载 osnet_x1_0 模型（Kaiyang Zhou 官方权重）
        self.model = torch.hub.load('kaiyangzhou/deep-person-reid', 'osnet_x1_0', pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __call__(self, img_np):
        if img_np is None or img_np.shape[0] == 0 or img_np.shape[1] == 0:
            return None
        img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
            features = F.normalize(features, dim=1)
        return features[0].cpu().numpy()