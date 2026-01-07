import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms

class MobileNetEmbedder:
    def __init__(self, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.model.classifier = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.cpu().numpy().flatten()
