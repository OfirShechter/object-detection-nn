import torch
from torch.utils.data import Dataset
from .utils.helpers import iou
import requests
import cv2
import numpy as np
from PIL import Image

# Create a dataset class to load the images and labels from the folder


class CocoDataset(Dataset):
    def __init__(
            self, coco_obj, categories, anchors, transform=None,
            image_size=416, grid_sizes=[13, 26, 52]
    ):
        self.coco = coco_obj
        self.cat_ids = self.coco.getCatIds(catNms=categories)
        # Get image ids for each category- should work with self.coco.getImgIds(catIds=cat_ids), but has a bug
        self.img_ids = [self.coco.getImgIds(
            catIds=cat_id) for cat_id in self.cat_ids]
        # Flatten the list
        self.img_ids = [item for sublist in self.img_ids for item in sublist]
        # self.img_ids = self.img_ids[:16]  # Limit to 16 images for testing

        # Image size
        self.image_size = image_size
        # Transformations
        self.transform = transform(
            image_size) if transform is not None else None
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2])
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes
        self.num_classes = len(categories)
        # Ignore IoU threshold
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.img_ids[idx]
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(
            imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        path = img_info['coco_url']

        response = requests.get(path, stream=True)
        if response.status_code == 200:
            img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(Image.fromarray(img))
        else:
            raise Exception(f"Failed to download image from {path}")

        bboxes = []
        category_ids = []
        for ann in anns:
            category_ids.append(ann["category_id"])
            bboxes.append(ann["bbox"])

        if self.transform is not None:
            augs = self.transform(
                image=img, bboxes=bboxes, category_ids=category_ids)
            img = augs["image"]
            bboxes = [[*bounderies, category_id] for bounderies,
                      category_id in zip(augs["bboxes"], augs["category_ids"])]

        # Normalize bounding box coordinates
        bboxes = [[x / self.image_size, y / self.image_size, w / self.image_size, h / self.image_size, label]
                  for x, y, w, h, label in bboxes]

        # Convert COCO bbox format (x, y, width, height) to (center_x, center_y, width, height)
        bboxes = [[x + w / 2, y + h / 2, w, h, label] for x, y, w, h, label in bboxes]
        
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label]
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6))
                   for s in self.grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = iou(torch.tensor(box[2:4]),
                              self.anchors,
                              is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            # At each scale, assigning the bounding box to the
            # best matching anchor box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative
                    # to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s, height * s)

                    # Idnetify the box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell,
                         height_cell]
                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 5] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                    # If the anchor box is already assigned, check if the
                    # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

                # Return the image and the target
        return img, tuple(targets)
