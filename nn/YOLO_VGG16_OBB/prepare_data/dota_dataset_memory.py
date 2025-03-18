import os
import torch
from torch.utils.data import Dataset
from ..utils.helpers import iou
from ..utils.constants import device
import cv2
import numpy as np
from PIL import Image
# Create a dataset class to load the images and labels from the folder


class DotaDataset(Dataset):
    def __init__(
            self, categories, anchors, transform=None, data_base_path=f"nn/dotadataset/train",
            image_size=416, grid_sizes=[13, 26, 52]
    ):
        self.images_path = f"{data_base_path}/images"
        self.labels_path = f"{data_base_path}/labelTxt-v1.0"
        self.img_ids = [os.path.splitext(f)[0] for f in os.listdir(
            self.images_path) if f.endswith('.png')]
        self.cat_ids_map = {category: i for i,
                            category in enumerate(categories)}
        self.img_ids = self.img_ids[:10]
        # Image size
        self.image_size = image_size
        # # Transformations
        self.transform = transform(
            image_size) if transform is not None else None
        # self.transform = None
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
        error_counter = 0
        while True:
            try:
                return self.getitem_helper(idx)
            except Exception as e:
                print(e)
                # choose random different idx
                idx = np.random.randint(0, len(self.img_ids))
                error_counter += 1
                if error_counter > 10:
                    print("Too many errors")
                    raise Exception("Too many errors")

    def getitem_helper(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.images_path, f"{img_id}.png")

        # Load image from memory
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(f"Failed to load image from {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(img))
        img_size_x = img.shape[1]
        img_size_y = img.shape[0]
        # Load labels
        label_path = os.path.join(self.labels_path, f"{img_id}.txt")
        if not os.path.exists(label_path):
            raise Exception(f"Labels file not found: {label_path}")

        bboxes = []
        angles = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue  # Skip invalid lines
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                category = parts[8]
                if category not in self.cat_ids_map:
                    raise Exception(f"Unknown category: {category}")
                class_label = self.cat_ids_map[category]

                # Convert OBB to (cx, cy, w, h, angle)
                poly = np.array([[x1, y1], [x2, y2], [x3, y3], [
                                x4, y4]], dtype=np.float32).reshape((-1, 1, 2))
                rect = cv2.minAreaRect(poly)
                (cx, cy), (w, h), angle = rect
                n_cx, n_cy, n_w, n_h = cx / img_size_x, cy / img_size_y, w / img_size_x, h / img_size_y
                if (n_cx < 0 or n_cy < 0 or n_w < 0 or n_h < 0) or (n_cx > 1 or n_cy > 1 or n_w > 1 or n_h > 1):
                    print('origin:', [x1, y1], [x2, y2], [x3, y3], [x4, y4])
                    print('poly', poly)
                    print('rect:', rect)
                    print('cx:', cx, 'cy:', cy, 'w:',
                          w, 'h:', h, 'angle:', angle)
                # else:
                #     print('No lower then 0:', n_cx, n_cy, n_w, n_h, angle)
                bboxes.append([n_cx, n_cy, n_w, n_h, class_label])
                rad_angle = np.deg2rad(angle)
                than_normelize = rad_angle / (np.pi / 2)
                angles.append(than_normelize)
        if self.transform is not None:
            augs = self.transform(
                image=img, bboxes=bboxes)
            img = augs["image"]
            bboxes = [[cx, cy, w, h, angle, class_label] for (
                cx, cy, w, h, class_label), angle in zip(augs["bboxes"], angles)]
        else:
            bboxes = [[cx, cy, w, h, angle, class_label]
                      for (cx, cy, w, h, class_label), angle in zip(bboxes, angles)]
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, angle, class_label]
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 7))
                   for s in self.grid_sizes]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = iou(torch.tensor(box[2:4], device=device),
                              self.anchors,
                              is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, angle, class_label = box

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
                         height_cell, angle]
                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:6] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 6] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                    # If the anchor box is already assigned, check if the
                    # IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

                # Return the image and the target
        return img, tuple(targets)
