
import os
from nn.YOLO_VGG16.utils.constants import ANCHORS
from nn.YOLO_VGG16.prepare_data.coco_dataset import CocoDataset
from nn.YOLO_VGG16.prepare_data.transforms import test_transform
from nn.YOLO_VGG16.utils.helpers import convert_cells_to_bboxes, get_coco_index_lable_map, load_checkpoint, nms, plot_image
from nn.YOLO_VGG16.utils.constants import device, s, leanring_rate, checkpoint_file
from nn.YOLO_VGG16.model.YOLO_VGG16_full import YOLO_VGG16_F
import torch
import torch.optim as optim
from nn.YOLO_VGG16.model.loss import YOLOLoss
from pycocotools.coco import COCO

class Object_Class_Detector():
    def __init__(self):
        model_path = f"./saved_models/vgg_f_modele32_vgg16_checkpoint.pth.tar"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        
        categories = ["dog"]
        load_model = True
        
        model = YOLO_VGG16_F(num_classes=len(categories)).to(device) 
        optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 
        scaler = torch.amp.GradScaler(device=device) 
        if load_model: 
            load_checkpoint(model_path, model, optimizer, leanring_rate, device) 

        model.eval()   
        self.model = model
        self.categories = categories

  
    def plot_marked_images(self, images):
        """
        Function to plot the images with bounding boxes
        """
        images = images.to(device)
        images = test_transform(images)
        marked_images = []
        with torch.no_grad(): 
            # Getting the model predictions 
            output = self.model(images) 
            # Getting the bounding boxes from the predictions 
            bboxes = [[] for _ in range(images.shape[0])] 
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
            nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6) 
            # Plotting the image with bounding boxes 
            image = plot_image(images[i].permute(1,2,0).detach().cpu(), nms_boxes, self.categories)
            marked_images.append(image)
        
        return marked_images

# %%

# %%
