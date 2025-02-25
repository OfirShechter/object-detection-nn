import torch
import torch.nn as nn
import torchvision.models as models

class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg16.features.children()))
        
        # YOLO-like detection head
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Final prediction layer (5 outputs: x, y, width, height, confidence)
        self.prediction_head = nn.Conv2d(64, 5, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.prediction_head(x)
        
        # Global average pooling to convert to (batch_size, 5)
        x = torch.mean(x, dim=[2, 3])
        
        return x