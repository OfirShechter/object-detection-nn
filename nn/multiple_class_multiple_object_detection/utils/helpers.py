import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Defining a function to calculate Intersection over Union (IoU)


def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(
            box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score

# Non-maximum suppression function to remove overlapping bounding boxes


def nms(bboxes_orig, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    bboxes = [box for box in bboxes_orig if box[1] > threshold]

    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_nms = []
    while len(bboxes) > 0:
        current_box = bboxes.pop(0)

        iou_intersections = [iou(
            torch.tensor(current_box[2:]), torch.tensor(bbox_nms[2:])
        ) for bbox_nms in bboxes_nms]
        max_iou = max(iou_intersections) if iou_intersections else 0
        if max_iou < iou_threshold:
            bboxes_nms.append(current_box)

    # Return bounding boxes after non-maximum suppression.
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
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

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
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()

# Function to plot images with bounding boxes and class labels

colors_options =  {
    'blue': (78, 179, 255),
    'pink': (255, 0, 145)
}
color_lables = ['blue', 'pink']

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
        score = box[1]
        box = box[2:]
        upper_left_x = int((box[0] - box[2] / 2) * w)
        upper_left_y = int((box[1] - box[3] / 2) * h)
        lower_right_x = int((box[0] + box[2] / 2) * w)
        lower_right_y = int((box[1] + box[3] / 2) * h)

        # Get color
        color = colors_options[color_lables[class_pred]]

        # Draw rectangle on image
        cv2.rectangle(img_drawn, (upper_left_x, upper_left_y),
                      (lower_right_x, lower_right_y), color, 2)

        # Put label text
        label = labels[class_pred]
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_drawn, (upper_left_x, upper_left_y - text_height - 10),
                      (upper_left_x + text_width, upper_left_y), color, -1)
        cv2.putText(img_drawn, label, (upper_left_x, upper_left_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(img_drawn, f"{score:.2f}", (upper_left_x, upper_left_y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if display:
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(img_drawn)  # Convert BGR to RGB for correct color display
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
