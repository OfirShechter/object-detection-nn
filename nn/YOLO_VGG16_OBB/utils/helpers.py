import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from .constants import device
# Defining a function to calculate Intersection over Union (IoU)


def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height, angle] format
        # Convert boxes to polygons
        polys1 = []
        polys2 = []

        # angle (box1[..., 4]) is in radian than- convert to degree
        angle1 = box1[..., 4] * (torch.pi / 2)
        angle2 = box2[..., 4] * (torch.pi / 2)
        angle1_degree = torch.rad2deg(angle1)
        angle2_degree = torch.rad2deg(angle2)

        for i in range(box1.shape[0]):
            poly1 = cv2.boxPoints(((box1[i, 0].item(), box1[i, 1].item(
            )), (box1[i, 2].item(), box1[i, 3].item()), angle1_degree[i].item()))
            poly2 = cv2.boxPoints(((box2[i, 0].item(), box2[i, 1].item(
            )), (box2[i, 2].item(), box2[i, 3].item()), angle2_degree[i].item()))
            polys1.append(poly1)
            polys2.append(poly2)

        # Convert polygons to torch tensors
        poly1 = torch.tensor(polys1, dtype=torch.float32)
        poly2 = torch.tensor(polys2, dtype=torch.float32)

        # Calculate intersection area
        inter_area = polygon_intersection_area(poly1, poly2)

        # Calculate union area
        box1_area = box1[..., 2] * box1[..., 3]
        box2_area = box2[..., 2] * box2[..., 3]
        union_area = box1_area + box2_area - inter_area

        # Calculate IoU score
        epsilon = 1e-6
        iou_score = inter_area / (union_area + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(
            box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = torch.tensor(box1[..., 0] * box1[..., 1], device=device)
        box2_area = torch.tensor(box2[..., 0] * box2[..., 1], device=device)
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


def polygon_intersection_area(poly1, poly2):
    # Ensure tensors are on CPU and convert to NumPy
    poly1_np = poly1.detach().cpu().numpy().astype(np.float32)
    poly2_np = poly2.detach().cpu().numpy().astype(np.float32)

    inter_areas = []
    for p1, p2 in zip(poly1_np, poly2_np):
        inter_poly = cv2.intersectConvexConvex(p1, p2)
        if inter_poly[0] > 0 and inter_poly[1] is not None:
            inter_areas.append(cv2.contourArea(inter_poly[1]))
        else:
            inter_areas.append(0.0)

    # Convert result back to a tensor
    return torch.tensor(inter_areas, dtype=torch.float32, device=device)


def nms(bboxes_orig, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    bboxes = [box for box in bboxes_orig if box[1] > threshold]

    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # Initialize the list of bounding boxes after non-maximum suppression.
    if (len(bboxes) > 0):
        first_box = bboxes.pop(0)
        bboxes_nms = [first_box]
    else:
        bboxes_nms = [max(bboxes_orig, key=lambda x: x[1])]

    while len(bboxes) >= 0:
        # Iterate over the remaining bounding boxes.
        for box in bboxes:
            # If the bounding boxes do not overlap or if the first bounding box has
            # a higher confidence, then add the second bounding box to the list of
            # bounding boxes after non-maximum suppression.
            if box[0] != first_box[0] or iou(
                    torch.tensor([first_box[2:]]),
                    torch.tensor([box[2:]]),
            ) < iou_threshold:
                # Check if box is not in bboxes_nms
                if box not in bboxes_nms:
                    # Add box to bboxes_nms
                    bboxes_nms.append(box)

        # Get the first bounding box.
        if len(bboxes) > 0:
            first_box = bboxes.pop(0)
        else:
            break

    # Return bounding boxes after non-maximum suppression.
    # box concist: [class_pred, score, x, y, width, height, angle]
    return bboxes_nms

# Function to convert cells to bounding boxes


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = predictions[..., 1:5]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(
            box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 6:], dim=-1).unsqueeze(-1)
        angle = torch.tanh(predictions[..., 5:6])
    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 6:7]
        angle = predictions[..., 5:6]

    # Calculate cell indices
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # Calculate x, y, width and height with proper scaling
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] +
                 cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]
    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 7) shape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height, angle), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 7)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()

# Function to plot images with bounding boxes and class labels


def plot_image(image, boxes, labels, display=True):
    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")

    # Convert image to NumPy array (if not already)
    img = np.array(image)
    h, w, _ = img.shape

    # Copy the image to avoid modifying the original
    img_drawn = img.copy()

    # Plot bounding boxes and labels
    for box in boxes:
        class_pred = int(box[0])
        cx, cy, bw, bh, angle = box[2:]

        # angle from rad_than back to degree
        angle_rad = angle * (np.pi / 2)
        angle = np.rad2deg(angle_rad)
        # Convert to absolute coordinates
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h

        # Get color
        color = colour_map(class_pred)
        # Get rotated rectangle
        rect = ((cx, cy), (bw, bh), angle)  # OpenCV expects angle in degrees
        box_points = cv2.boxPoints(rect)  # Get corner points
        box_points = np.int32(box_points)  # Convert to integer

        # Draw the rotated rectangle
        cv2.polylines(img_drawn, [box_points],
                      isClosed=True, color=color, thickness=2)

        # Put label text near the rectangle
        # label = labels[class_pred]
        # (text_width, text_height), baseline = cv2.getTextSize(
        #     label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # text_x, text_y = int(cx - text_width / 2), int(cy - bh / 2 - 10)
        # cv2.rectangle(img_drawn, (text_x, text_y - text_height - 4),
        #               (text_x + text_width, text_y), color, -1)
        # cv2.putText(img_drawn, label, (text_x, text_y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if display:
        # Display the image
        plt.figure(figsize=(8, 6))
        # Convert BGR to RGB
        plt.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return img_drawn  # Return the modified image with drawn bounding boxes


def get_coco_index_lable_map(coco, lables):
    coco_id_to_name = {coco.getCatIds(
        catNms=lable)[0]: lable for lable in lables}
    return coco_id_to_name

# Function to save checkpoint


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# Function to load checkpoint


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
