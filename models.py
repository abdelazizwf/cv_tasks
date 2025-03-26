from functools import partial

from torch import nn
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ConvBlock(nn.Module):
    
    def __init__(self, inp, l1, l2):
        super().__init__()
        
        conv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding="same")
        
        self.layers = nn.Sequential(
            conv2d(inp, l1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(l1),
            conv2d(l1, l2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(l2),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.layers(x)


class ClassificationNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3, 8, 16),
            ConvBlock(16, 32, 64),
            ConvBlock(64, 128, 256),
            nn.AvgPool2d(16),
            nn.Flatten(),
            nn.Linear(256, 5)
        )
    
    def forward(self, X):
        return self.layers(X)


class ClassificationTLNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = models.resnet50(weights="DEFAULT")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)
    
    def forward(self, X):
        return self.model(X)


class DetectionTLNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    
    def forward(self, images, labels_and_boxes=None):
        if self.training:
            assert labels_and_boxes is not None
            return self.model(images, labels_and_boxes)
        else:
            return self.model(images)


class SiameseNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(1, 8, 16),
            nn.LocalResponseNorm(3),
            ConvBlock(16, 32, 64),
            ConvBlock(64, 128, 256),
            nn.AvgPool2d(32),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.Dropout(),
            nn.Linear(512, 128)
        )

    def f_pass(self, X):
        return self.layers(X)
    
    def forward(self, X1, X2):
        out1 = self.f_pass(X1)
        out2 = self.f_pass(X2)
        return out1, out2
