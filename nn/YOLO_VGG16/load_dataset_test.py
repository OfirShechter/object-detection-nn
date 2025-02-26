#%%
import sys
import os

# Add the root directory of your project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f"Adding {project_root} to PYTHONPATH")
sys.path.append(project_root)

from nn.YOLO_VGG16.utils.helpers import convert_cells_to_bboxes, get_coco_index_lable_map, nms, plot_image
from nn.YOLO_VGG16.utils.constants import ANCHORS
from nn.YOLO_VGG16.coco_dataset import CocoDataset
from nn.YOLO_VGG16.transforms import test_transform
import torch
from pycocotools.coco import COCO

#%%
coco = COCO('../cocodataset/annotations/instances_train2017.json')
categories = ["dog"]
id_to_lable = get_coco_index_lable_map(coco, categories)

#%%
dataset = CocoDataset( 
	coco_obj=coco, 
	categories=categories,
	grid_sizes=[13, 26, 52], 
	anchors=ANCHORS, 
	transform=test_transform 
) 


#%%
# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# Defining the grid size and the scaled anchors 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
	1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

#%%
# Getting a batch from the dataloader 
x, y = next(iter(loader)) 

#%%
# Getting the boxes coordinates from the labels 
# and converting them into bounding boxes without scaling 
boxes = [] 
for i in range(y[0].shape[1]): 
	anchor = scaled_anchors[i] 
	boxes += convert_cells_to_bboxes( 
			y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
			)[0] 

# Applying non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 

# Plotting the image with the bounding boxes 
plot_image(x[0].permute(1,2,0).to("cpu"), boxes, id_to_lable)

# %%
