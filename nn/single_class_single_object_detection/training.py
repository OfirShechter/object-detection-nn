import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from .detection_model import DetectionModel
from ..cocodataset.code.torch_dataset import CocoDetectionByURL
from ..utils.torch_general_utils import load_to_model, save_model

def main():
    saved_model_path = 'nn/single_class_single_object_detection/saved_models/scso_detection_model.pth'
    dataset = CocoDetectionByURL(annFile='nn/cocodataset/annotations/instances_train2017.json', categories=['dog'])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

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
        for batch_count, (images, targets) in enumerate(dataloader, 1):
            print(f"~~~~~~~~~ Running Batch {batch_count} ~~~~~~~~~~")
            images = images.to(device)
            targets = targets.to(device)  # x, y, w, h, confidence
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Batch loss: {loss.item()}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
        save_model(model, saved_model_path)
    
    print("Training complete")

if __name__ == '__main__':
    main()
