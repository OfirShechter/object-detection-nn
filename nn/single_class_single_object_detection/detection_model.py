import torch
import torch.nn as nn
import torchvision.models as models

class DetectionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(DetectionModel, self).__init__()
        # Initial high-resolution processing (before VGG-16)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Larger receptive field
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling to reduce input size
        )
        
        # Load pretrained VGG16
        vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg16.features.children()))
        
        # YOLO-like detection head
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
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
        print(f"Input shape: {x.shape}")
        self.initial_conv(x)
        print(f"Initial conv output shape: {x.shape}")
        x = self.backbone(x)
        print(f"Backbone output shape: {x.shape}")
        x = self.conv1(x)
        print(f"Conv1 output shape: {x.shape}")
        x = self.relu1(x)
        print(f"Relu1 output shape: {x.shape}")
        x = self.conv2(x)
        print(f"Conv2 output shape: {x.shape}")
        x = self.relu2(x)
        print(f"Relu2 output shape: {x.shape}")
        x = self.conv3(x)
        print(f"Conv3 output shape: {x.shape}")
        x = self.relu3(x)
        print(f"Relu3 output shape: {x.shape}")
        x = self.prediction_head(x)
        print(f"Prediction head output shape: {x.shape}")
        
        # Global average pooling to convert to (batch_size, 5)
        x = torch.mean(x, dim=[2, 3])
        
        return x