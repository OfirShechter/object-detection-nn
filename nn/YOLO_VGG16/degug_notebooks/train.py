#%%
# !git checkout part_2
# !pwd
#%%
remote_mode = True

import sys
import os

if remote_mode:
    
    # print working directory and change to another
    print(f"Current working directory: {os.getcwd()}")
    # os.chdir('Rar')
    # os.chdir('object-detection-nn')
    print(f"Current working directory: {os.getcwd()}")

    # Add the root directory of your project to the PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
else:
    # Add the root directory of your project to the PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Adding {project_root} to PYTHONPATH")
sys.path.append(project_root)


from nn.YOLO_VGG16 import training_loop
from nn.YOLO_VGG16.utils.constants import ANCHORS
from nn.YOLO_VGG16.prepare_data.coco_dataset import CocoDataset
from nn.YOLO_VGG16.prepare_data.transforms import train_transform
from nn.YOLO_VGG16.utils.helpers import get_coco_index_lable_map, load_checkpoint, save_checkpoint
from nn.YOLO_VGG16.utils.constants import device, s, leanring_rate, save_model, epochs, checkpoint_file
from nn.YOLO_VGG16.model.YOLO_v3 import YOLOv3
import torch
import torch.optim as optim
from nn.YOLO_VGG16.model.loss import YOLOLoss
from pycocotools.coco import COCO
from tqdm import tqdm

#%%
# Creating the model from YOLOv3 class 
load_model = True
model = YOLOv3().to(device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.amp.GradScaler(device=device) 
model_path = f"/home/dcor/niskhizov/Rar/object-detection-nn/nn/YOLO_VGG16/degug_notebooks/{checkpoint_file}"
# Loading the checkpoint 
if load_model: 
    load_checkpoint(model_path, model, optimizer, leanring_rate, device) 

#%%
coco = COCO('/home/dcor/niskhizov/Rar/object-detection-nn/nn/YOLO_VGG16/degug_notebooks/temp/instances_train2017.json')
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

#%%
epochs = 1000000000000000000000
# Training the model 
for e in range(1, epochs+1): 
	print("Epoch:", e) 
    ################# dos
    	# Creating a progress bar 
	progress_bar = tqdm(train_loader, leave=True) 

	# Initializing a list to store the losses 
	losses = [] 

	# Iterating over the training data 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(device) 
		y0, y1, y2 = ( 
			y[0].to(device), 
			y[1].to(device), 
			y[2].to(device), 
		) 

		with torch.amp.autocast(device_type=device): 
			# Getting the model predictions 
			outputs = model(x) 
			# Calculating the loss at each scale 
			loss = ( 
				loss_fn(outputs[0], y0, scaled_anchors[0]) 
				+ loss_fn(outputs[1], y1, scaled_anchors[1]) 
				+ loss_fn(outputs[2], y2, scaled_anchors[2]) 
			) 

		# Add the loss to the list 
		losses.append(loss.item()) 

		# Reset gradients 
		optimizer.zero_grad() 

		# Backpropagate the loss 
		scaler.scale(loss).backward() 

		# Optimization step 
		scaler.step(optimizer) 

		# Update the scaler for next iteration 
		scaler.update() 

		# update progress bar with loss 
		mean_loss = sum(losses) / len(losses) 
		progress_bar.set_postfix(loss=mean_loss)

    #################
	# training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 

	# Saving the model 
	if save_model: 
		save_checkpoint(model, optimizer, filename=model_path)

# %%
