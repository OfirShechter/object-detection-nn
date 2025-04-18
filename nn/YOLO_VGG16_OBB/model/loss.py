import torch
import torch.nn as nn

from ..utils.helpers import iou

# Defining YOLO loss class


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.than = nn.Tanh()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = 10 * self.bce(
            self.sigmoid(pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )
        
        # reg_object_loss = 10 * self.bce(
        #     self.sigmoid(pred[..., 0:1][obj]), (target[..., 0:1][obj]),
        # )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        pred_angle = torch.tanh(pred[..., 5])
        # Box prediction confidence
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]),
                               torch.exp(pred[..., 3:5]) *
                               anchors, pred_angle.unsqueeze(-1)
                               ], dim=-1)
        # Calculating intersection over union for prediction and target
        ious = iou(box_preds[obj], target[..., 1:6][obj]).detach()
        # Calculating Object loss
        object_loss = 10 * self.mse(self.sigmoid(pred[..., 0:1][obj]),
                               ious * target[..., 0:1][obj]).clamp(0, 10)

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss

        box_loss = self.mse(pred[..., 1:5][obj],
                            target[..., 1:5][obj])

        # Calculate angle loss
        angle_loss = 10 * self.mse(pred_angle[obj],
                              target[..., 5][obj])
        # Claculating class loss
        class_loss = self.cross_entropy((pred[..., 6:][obj]),
                                        target[..., 6][obj].long())

        # print('~~~~~~~~~~~~~ FIND UNCAPPED ~~~~~~~~~~~~~')
        # print('box_loss:', box_loss)
        # print('angle_loss:', angle_loss)
        # print('object_loss:', object_loss)
        # print('no_object_loss:', no_object_loss)
        # print('class_loss:', class_loss)
        # print('reg_object_loss', reg_object_loss)

        # Total loss
        return (
            box_loss
            + angle_loss
            + object_loss
            + no_object_loss
            + class_loss
            # + reg_object_loss
        )
