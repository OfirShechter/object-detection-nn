#%%
import torch
import torch.nn as nn
import torchvision.models as models
#%%
# Load pretrained VGG16 (with batch normalization)
vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
vgg16.features
# %%
vgg16.features[:24]
# %%
vgg16.features[24:34]
# %%
vgg16.features[34:]

# %%
