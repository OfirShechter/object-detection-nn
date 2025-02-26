from nn.YOLO_VGG16 import training_loop
from .utils.constants import ANCHORS
from .prepare_data.coco_dataset import CocoDataset
from .prepare_data.transforms import test_transform
from .utils.helpers import convert_cells_to_bboxes, get_coco_index_lable_map, load_checkpoint, nms, plot_image, save_checkpoint
from .utils.constants import device, s, leanring_rate, save_model, epochs, checkpoint_file
from .model.YOLO_v3 import YOLOv3
import torch
import torch.optim as optim
from .model.loss import YOLOLoss
from pycocotools.coco import COCO

# Setting the load_model to True 
load_model = True
  
# Defining the model, optimizer, loss function and scaler 
model = YOLOv3().to(device) 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 
  
# Loading the checkpoint 
if load_model: 
    load_checkpoint(checkpoint_file, model, optimizer, leanring_rate) 

#%%
coco = COCO('nn/cocodataset/annotations/instances_val2017.json')
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

# Defining the train data loader 
test_loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# Getting a sample image from the test data loader 
x, y = next(iter(test_loader)) 
x = x.to(device) 
  
model.eval() 
with torch.no_grad(): 
    # Getting the model predictions 
    output = model(x) 
    # Getting the bounding boxes from the predictions 
    bboxes = [[] for _ in range(x.shape[0])] 
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
model.train() 
  
# Plotting the image with bounding boxes for each image in the batch 
for i in range(batch_size): 
    # Applying non-max suppression to remove overlapping bounding boxes 
    nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6) 
    # Plotting the image with bounding boxes 
    plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, id_to_lable)
