#%%
import sys
import os

# Add the root directory of your project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f"Adding {project_root} to PYTHONPATH")
sys.path.append(project_root)

# Now you can use relative imports
from nn.utils.torch_general_utils import load_to_model, save_model
from nn.single_class_single_object_detection.detection_model import DetectionModel

os.getcwd()
#%%
from detection_model import DetectionModel

from pycocotools.coco import COCO
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from PIL import Image
#%%
os.listdir()
# %%
# Load the COCO dataset
coco = COCO('../cocodataset/annotations/instances_train2017.json')
# %%
# Get all category IDs (e.g., person, car, etc.)
cat_ids = coco.getCatIds(catNms=['dog'])  # Change to any category
img_ids = coco.getImgIds(catIds=cat_ids)
img_ids = [98304, 204800, 524291, 311301, 491525, 147471, 131087, 278550, 581654, 253981]
# %%
# Load and display an image
img_info = coco.loadImgs(img_ids[8])[0]
img_url = img_info['coco_url']
img_info
# %%
def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img
    else:
        print("Failed to download image")
        return None

# Load and show the image
img = load_image_from_url(img_url)

if img is not None:
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
# %%
# Load annotations for the selected image
ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
annotations = coco.loadAnns(ann_ids)
annotations

#%%
# Draw bounding boxes
for ann in annotations:
    x, y, w, h = ann['bbox']
    boxed_img = np.array(img)
    cv2.rectangle(boxed_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(boxed_img)
plt.axis('off')
plt.show()

# %%
saved_model_path = './saved_models/scso_detection_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"##################### Using device {device} #####################")
model = DetectionModel().to(device)
load_to_model(model, saved_model_path, device)
trasform = models.VGG16_Weights.DEFAULT.transforms()
# %%
tensor_img = trasform(img)

target = model(tensor_img.unsqueeze(0).to(device))
# %%
# Draw bounding boxes
for t in target:
    x, y, w, h, confidance = t
    boxed_img = np.array(img)
    cv2.rectangle(boxed_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(boxed_img)
plt.axis('off')
plt.show()

# %%
from torchvision import models
import torch.nn as nn
vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
nn.Sequential(*list(vgg16.features.children()))
# %%
