import torch
import json
import os
import config
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.patches as patches
from matplotlib import pyplot as plt


#############################
#       Loss Function       #
#############################
class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, p, a):
        # Calculate IOU of each box
        iou = get_iou(p, a)                     # (batch, S, S, B, B)
        print(torch.sum(iou < 0.0).item())
        print(torch.sum(iou > 1.0).item())
        max_iou = torch.max(iou, dim=-1)[0]     # (batch, S, S, B)

        # Get masks
        bbox_mask = bbox_attr(a, 4) > 0
        obj_ij = torch.zeros_like(bbox_mask).scatter_(          # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),        # (batch, S, S, B)
            value=1
        )
        noobj_ij = ~obj_ij                      # Opposite, 1's if grid I, bbox J NOT responsible for prediction
        obj_i = bbox_mask[:, :, :, 0:1]         # 1 if grid I has any object at all

        # XY position losses
        x_losses = F.mse_loss(
            obj_ij * bbox_attr(p, 0),
            obj_ij * bbox_attr(a, 0),
            reduction='sum'
        )
        y_losses = F.mse_loss(
            obj_ij * bbox_attr(p, 1),
            obj_ij * bbox_attr(a, 1),
            reduction='sum'
        )
        pos_losses = x_losses + y_losses
        print('pos_losses', pos_losses.item())

        # Bbox dimension losses
        width_losses = F.mse_loss(
            obj_ij * torch.sqrt(torch.clamp(bbox_attr(p, 2), min=config.EPSILON)),
            obj_ij * torch.sqrt(torch.clamp(bbox_attr(a, 2), min=config.EPSILON)),
            reduction='sum'
        )
        height_losses = F.mse_loss(
            obj_ij * torch.sqrt(torch.clamp(bbox_attr(p, 3), min=config.EPSILON)),
            obj_ij * torch.sqrt(torch.clamp(bbox_attr(a, 3), min=config.EPSILON)),
            reduction='sum'
        )
        dim_losses = width_losses + height_losses
        print('dim_losses', dim_losses.item())
        # print(torch.sum(bbox_attr(p, 2) < 0).item())
        # print(torch.sum(bbox_attr(p, 3) < 0).item())
        # print(torch.sum(bbox_attr(a, 2) < 0).item())
        # print(torch.sum(bbox_attr(a, 3) < 0).item())

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = F.mse_loss(
            obj_ij * bbox_attr(p, 4),
            obj_ij * max_iou,
            reduction='sum'
        )
        print('obj_confidence_losses', obj_confidence_losses.item())
        noobj_confidence_losses = F.mse_loss(
            noobj_ij * bbox_attr(p, 4),
            0.0 * max_iou,
            reduction='sum'
        )
        print('noobj_confidence_losses', noobj_confidence_losses.item())

        # Classification losses
        class_losses = F.mse_loss(
            obj_i * F.softmax(p[:, :, :, 5*config.B:], dim=3),
            obj_i * F.softmax(a[:, :, :, 5*config.B:], dim=3),
            reduction='sum'
        )
        print('class_losses', class_losses.item())

        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        return total / config.BATCH_SIZE


#################################
#       Helper Functions        #
#################################
def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = br - tl
    intersection = intersection_sides[:, :, :, :, :, 0] \
                   * intersection_sides[:, :, :, :, :, 1]       # (batch, S, S, B, B)

    # Clamp intersection to be non-negative
    intersection = torch.clamp(intersection, min=0.0)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero and negative unions
    invalid_unions = (intersection > union)
    union[invalid_unions] = config.EPSILON
    intersection[invalid_unions] = 0.0

    return intersection / union


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    return data[:, :, :, i:5*config.B:5]


def plot_boxes(data, labels, classes, threshold=0.5):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S

    fig, ax = plt.subplots()
    plt.imshow(data.permute(1, 2, 0))
    for i in range(labels.size(dim=0)):
        for j in range(labels.size(dim=1)):
            for k in range(config.B):
                bbox = labels[i, j, 5*k:5*(k+1)]
                confidence = bbox[4].item()
                if confidence > threshold:
                    class_index = torch.argmax(labels[i, j, -config.C:]).item()
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    bbox_tl = (
                        bbox[0] * config.IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                        bbox[1] * config.IMAGE_SIZE[1] + i * grid_size_y - height / 2
                    )
                    rect = patches.Rectangle(
                        bbox_tl,
                        width,
                        height,
                        facecolor='none',
                        linewidth=1,
                        edgecolor='orange'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        bbox_tl[0],
                        bbox_tl[1],
                        f'{classes[class_index]} {round(confidence * 100, 1)}%',
                        bbox=dict(facecolor='orange', edgecolor='none'),
                        fontsize=6
                    )
    plt.show()
