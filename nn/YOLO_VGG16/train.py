from nn.YOLO_VGG16 import training_loop
from .utils.constants import ANCHORS
from .prepare_data.coco_dataset import CocoDataset
from .prepare_data.transforms import train_transform
from .utils.helpers import get_coco_index_lable_map, save_checkpoint
from .utils.constants import device, s, leanring_rate, save_model, epochs, checkpoint_file
from .model.YOLO_v3 import YOLOv3
import torch
import torch.optim as optim
from .model.loss import YOLOLoss
from pycocotools.coco import COCO

# Creating the model from YOLOv3 class 
model = YOLOv3().to(device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.amp.GradScaler(device=device) 

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
	transform=train_transform 
) 

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# Scaling the anchors 
scaled_anchors = ( 
	torch.tensor(ANCHORS) *
	torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 

# Training the model 
for e in range(1, epochs+1): 
	print("Epoch:", e) 
	training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 

	# Saving the model 
	if save_model: 
		save_checkpoint(model, optimizer, filename=checkpoint_file)
