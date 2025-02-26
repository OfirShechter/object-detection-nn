#%%
import sys
import os

# Add the root directory of your project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f"Adding {project_root} to PYTHONPATH")
sys.path.append(project_root)

from nn.YOLO_VGG16.model.YOLO_v3 import YOLOv3
import torch

#%%
# Setting number of classes and image size 
num_classes = 1
IMAGE_SIZE = 416

# Creating model and testing output shapes 
model = YOLOv3(num_classes=num_classes) 
x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
out = model(x) 
print(out[0].shape) 
print(out[1].shape) 
print(out[2].shape) 

# Asserting output shapes 
assert model(x)[0].shape == (1, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) 
assert model(x)[1].shape == (1, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5) 
assert model(x)[2].shape == (1, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5) 
print("Output shapes are correct!")

# %%
