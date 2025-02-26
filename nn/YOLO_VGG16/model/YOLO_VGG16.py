import torch
import torch.nn as nn

from .layers import CNNBlock, ResidualBlock, ScalePrediction

# Class for defining YOLOv3 model 
class yolo_vgg16(nn.Module): 
	def __init__(self, in_channels=3, num_classes=20): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		# Layers list for YOLOv3 
		self.layers = nn.ModuleList([ 
			CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
			CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(64, num_repeats=1), 
			CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(128, num_repeats=2), 
			CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(256, num_repeats=8), 
			CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(512, num_repeats=8), 
			CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(1024, num_repeats=4), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(1024, use_residual=False, num_repeats=1), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(512, num_classes=num_classes), 
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			nn.Upsample(scale_factor=2), 
			CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
			CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(512, use_residual=False, num_repeats=1), 
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(256, num_classes=num_classes), 
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
			nn.Upsample(scale_factor=2), 
			CNNBlock(384, 128, kernel_size=1, stride=1, padding=0), 
			CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(256, use_residual=False, num_repeats=1), 
			CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(128, num_classes=num_classes) 
		]) 
	
	# Forward pass for YOLOv3 with route connections and scale predictions 
	def forward(self, x): 
		outputs = [] 
		route_connections = [] 

		for layer in self.layers: 
			if isinstance(layer, ScalePrediction): 
				outputs.append(layer(x)) 
				continue
			x = layer(x) 

			if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: 
				route_connections.append(x) 
			
			elif isinstance(layer, nn.Upsample): 
				x = torch.cat([x, route_connections[-1]], dim=1) 
				route_connections.pop() 
		return outputs
