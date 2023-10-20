from ultralytics import YOLO
import torch.nn as nn
import torch


class FeatureExtractionModel:
    def __init__(self, weights_path='ClassWeights.pt', device='cpu'):
        self.device = device
        self.model = YOLO(weights_path).model.model.to(device)
        self.model[-1].linear = nn.Identity()

    def extract_features(self,mask):
        return self.model(mask.float().unsqueeze(0))
