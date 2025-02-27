import torch
import torch.nn as nn
import torchvision.models as models
from .layers import CNNBlock, ResidualBlock, ScalePrediction

# Class for defining YOLOv3 model 
class yolo_vgg16(nn.Module): 
	def __init__(self, in_channels=3, num_classes=20): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels
		
		vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
		backbone = vgg16.features
  
		self.large_features_layer = backbone[:24] # up to conv4_3. 
		self.medium_features_layer = backbone[24:34] # up to conv5_3. 
		self.small_features_layer = backbone[34:] # final

		self.large_object_detection_layers = nn.ModuleList([
      		CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(1024, use_residual=False, num_repeats=1), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(512, num_classes=num_classes), 
		])
  
		self.medium_object_detection_layers = nn.ModuleList([
      		CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			nn.Upsample(scale_factor=2), 
			CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
			CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(512, use_residual=False, num_repeats=1), 
			CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(256, num_classes=num_classes), 
		])
      
		self.small_object_detection_layers = nn.ModuleList([ 
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
		
		# Feature Extraction from VGG16
		x_large = self.large_features_layer(x)  # conv4_3 (shallow features)
		x_medium = self.medium_features_layer(x_large)  # conv5_3 (deeper features)
		x_small = self.small_features_layer(x_medium)  # Final feature extractor output


		for layer in self.large_object_detection_layers: 
			if isinstance(layer, ScalePrediction):
				outputs.append(layer(x_large))
				continue
			x_large = layer(x_large)
                
                
			# if isinstance(layer, ScalePrediction): 
			# 	outputs.append(layer(x)) 
			# 	continue
			# x = layer(x) 

			# if isinstance(layer, ResidualBlock) and layer.num_repeats == 8: 
			# 	route_connections.append(x) 
			
			# elif isinstance(layer, nn.Upsample): 
			# 	x = torch.cat([x, route_connections[-1]], dim=1) 
			# 	route_connections.pop() 
		return outputs
