#%%
# !git pull
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
from nn.YOLO_VGG16.utils.helpers import convert_cells_to_bboxes, get_coco_index_lable_map, load_checkpoint, nms, plot_image, save_checkpoint
from nn.YOLO_VGG16.utils.constants import device, s, leanring_rate, save_model, epochs, checkpoint_file
from nn.YOLO_VGG16.model.YOLO_VGG16 import YOLO_VGG16
import torch
import torch.optim as optim
from nn.YOLO_VGG16.model.loss import YOLOLoss
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

if remote_mode:
    model_path_base = f"/home/dcor/niskhizov/Rar/object-detection-nn/nn/YOLO_VGG16/degug_notebooks/"
    coco_path = '/home/dcor/niskhizov/Rar/object-detection-nn/nn/YOLO_VGG16/degug_notebooks/temp/instances_train2017.json'
else:
    model_path_base = f""
    coco_path = '../../cocodataset/annotations/instances_train2017.json'

#%%
coco = COCO(coco_path)
categories = ["dog"]
id_to_lable = get_coco_index_lable_map(coco, categories)

#%%
# Creating the model from YOLOv3 class 
load_model = False
model = YOLO_VGG16(num_classes=len(categories)).to(device) 

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = leanring_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.amp.GradScaler(device=device) 
# Loading the checkpoint 
if load_model: 
    load_checkpoint(model_path_base + f"vgg16_{checkpoint_file}", model, optimizer, leanring_rate, device) 

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/YOLO_VGG16')


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
epochs = 10
# Training the model 
for e in range(1, epochs+1): 
	print("Epoch:", e) 
    ################# dos
    	# Creating a progress bar 
	progress_bar = tqdm(train_loader, leave=True) 

	# Initializing a list to store the losses 
	losses = [] 

	# Iterating over the training data 
	for batch_idx, (x, y) in enumerate(progress_bar): 
		x = x.to(device) 
		y0, y1, y2 = ( 
			y[0].to(device), 
			y[1].to(device), 
			y[2].to(device), 
		) 
		print(f"target shape: {y0.shape}, {y1.shape}, {y2.shape}")

		with torch.amp.autocast(device_type=device): 
			# Getting the model predictions 
			outputs = model(x) 
			print(f"output shape: {outputs[0].shape}, {outputs[1].shape}, {outputs[2].shape}")
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
  
        # Log the loss to TensorBoard
		writer.add_scalar('Loss/train', mean_loss, e * len(train_loader) + batch_idx)

		# Log images to TensorBoard every 100 batches
		if batch_idx % 100 == 0:
			model.eval()
			with torch.no_grad():
				output = model(x)
				bboxes = [[] for _ in range(x.shape[0])]
				for i in range(3):
					batch_size, A, S, _, _ = output[i].shape
					anchor = scaled_anchors[i]
					boxes_scale_i = convert_cells_to_bboxes(output[i], anchor, s=S, is_predictions=True)
					for idx, box in enumerate(boxes_scale_i):
						bboxes[idx] += box
				for i in range(batch_size):
					nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6)
					img_with_boxes = plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes, id_to_lable)
					img_with_boxes = T.ToTensor()(img_with_boxes)
					writer.add_image(f'Train/Image_{e}_{batch_idx}', img_with_boxes, e * len(train_loader) + batch_idx)
			model.train()
   
	# Saving the model 
	if save_model: 
		save_checkpoint(model, optimizer, filename=model_path_base +f"{e}_vgg16_checkpoint.pth.tar")
		# delete checkpoint of previous 2 batch_idx if exists
		if e >= 2:
			os.remove(model_path_base + f"{e}_vgg16_checkpoint.pth.tar")


    #################
	# training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 


# %%
