import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os

# Load pre-trained VGG16 model
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.eval()  # Set to evaluation mode

# Define image transformation (resize, normalize, convert to tensor)
transform = models.VGG16_Weights.DEFAULT.transforms()

images_base_path = "backbone/images"

# Download ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# Perform inference on each image
for image_path in os.listdir(images_base_path):
    print(image_path)
    # Load image
    img = Image.open(f"{images_base_path}/{image_path}")

    # Preprocess image
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get top predicted class
    predicted_class = torch.argmax(output[0]).item()
    predicted_label = labels[predicted_class]

    # Display image and prediction
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_label}")
    plt.show()
