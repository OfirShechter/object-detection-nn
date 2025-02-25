import torch
import torch.nn as nn
import torchvision.models as models

class DetectionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(DetectionModel, self).__init__()
        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        
        # Freeze VGG16 layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Custom detection head
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Output layers for bounding box coordinates and class scores
        self.bbox = nn.Conv2d(64, 4, kernel_size=1)  # 4 coordinates for AABB
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)  # 1 class (basketball)
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        bbox = self.bbox(x)
        cls = self.classifier(x)
        
        return bbox, cls