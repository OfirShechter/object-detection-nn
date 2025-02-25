import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from .detection_model import DetectionModel
from ..cocodataset.code.torch_dataset import CocoDetectionByURL

dataset = CocoDetectionByURL(annFile='nn/cocodataset/annotations/instances_train2017.json', categories=['dog'])
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"##################### Using device {device} #####################")
model = DetectionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = images.to(device)
        targets =targets.to(device)  # x, y, w, h, confidence
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch loss: {loss.item()}")
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
