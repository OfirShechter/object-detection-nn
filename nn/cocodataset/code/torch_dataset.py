import torch
from torch.utils.data import Dataset
from torchvision import models
from pycocotools.coco import COCO
import requests
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class CocoDetectionByURL(Dataset):
    def __init__(self, annFile, categories, allow_multiple_obj=False):
        self.coco = COCO(annFile)
        self.cat_ids = self.coco.getCatIds(catNms=categories)
        img_ids = self.coco.getImgIds(catIds=self.cat_ids)[:10] # Limit to 16 images for testing
        # fillter images with multiple annotations
        if not allow_multiple_obj:
            self.img_ids = [img_id for img_id in img_ids if len(self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)) == 1]
        print(f"ImageIds: {img_ids}")
        
        # Define image transformation (Resize + Normalize)
        image_size = 512
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to match model input
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG-16 Normalization
        ])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.img_ids[index]
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        path = img_info['coco_url']

        response = requests.get(path, stream=True)
        if response.status_code == 200:
            img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            raise Exception(f"Failed to download image from {path}")
        
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor([ann['bbox'] + [1.0] for ann in anns], dtype=torch.float32).squeeze()  # x, y, w, h, confidence
        return img, target

    def __len__(self):
        return len(self.img_ids)