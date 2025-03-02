
import os

import cv2
import numpy as np
from nn.YOLO_VGG16.utils.constants import ANCHORS
from nn.YOLO_VGG16.prepare_data.transforms import execute_transform
from nn.YOLO_VGG16.utils.helpers import convert_cells_to_bboxes, load_checkpoint, nms, plot_image
from nn.YOLO_VGG16.utils.constants import device, s, leanring_rate, image_size
from nn.YOLO_VGG16.model.YOLO_VGG16_full import YOLO_VGG16_F
import torch
import torch.optim as optim
from PIL import Image

class Object_Class_Detector():
    def __init__(self, model_path=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        
        categories = ["dog"]
        load_model = True
        
        model = YOLO_VGG16_F(num_classes=len(categories)).to(device) 
        optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 
        if load_model: 
            load_checkpoint(model_path, model, optimizer, leanring_rate, device) 

        model.eval()   
        self.model = model
        self.categories = categories
        self.transform = execute_transform(image_size=image_size)

  
    def plot_marked_images(self, images):
        """
        Function to plot the images with bounding boxes
        """
        # Convert the image from BGR to RGB
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        
        # Convert the images to a NumPy array and apply the test transform
        images_np = np.array([np.array(Image.fromarray(img)) for img in images_rgb])
        transformed_images = [self.transform(image=img)['image'] for img in images_np]
        transformed_images = torch.stack(transformed_images).to(device)

        marked_images = []
        with torch.no_grad(): 
            # Getting the model predictions 
            output = self.model(transformed_images) 
            # Getting the bounding boxes from the predictions 
            bboxes = [[] for _ in range(transformed_images.shape[0])] 
            anchors = ( 
                    torch.tensor(ANCHORS) 
                        * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
                    ).to(device) 
        
            # Getting bounding boxes for each scale 
            for i in range(3): 
                batch_size, A, S, _, _ = output[i].shape 
                anchor = anchors[i] 
                boxes_scale_i = convert_cells_to_bboxes( 
                                    output[i], anchor, s=S, is_predictions=True
                                ) 
                for idx, (box) in enumerate(boxes_scale_i): 
                    bboxes[idx] += box 
        
        # Plotting the image with bounding boxes for each image in the batch 
        for i in range(batch_size): 
            # Applying non-max suppression to remove overlapping bounding boxes 
            nms_boxes = nms(bboxes[i], iou_threshold=0, threshold=0.6) 
            # Plotting the image with bounding boxes 
            image = plot_image(transformed_images[i].permute(1,2,0).detach().cpu(), nms_boxes, self.categories)
            marked_images.append(image)
        
        return marked_images
