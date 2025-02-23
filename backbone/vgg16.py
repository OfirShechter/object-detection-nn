import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.eval()  # Set to evaluation mode

# Define image transformation (resize, normalize, convert to tensor)
transform = models.VGG16_Weights.DEFAULT.transforms

# Load a few sample images from URLs
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Push_van_cat.jpg/640px-Push_van_cat.jpg",  # Cat
    "https://upload.wikimedia.org/wikipedia/commons/9/99/Golden_Retriever_medium-to-long_coat.jpg",  # Dog
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Big_Ben_Clock_Tower_-_London%2C_England_-_April_2009.jpg/640px-Big_Ben_Clock_Tower_-_London%2C_England_-_April_2009.jpg"  # Landmark
]

# Download ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# Perform inference on each image
for image_url in image_urls:
    # Load image
    response = requests.get(image_url, stream=True)
    img = Image.open(response.raw).convert("RGB")

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
