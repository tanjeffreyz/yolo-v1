import torch
import json
import os
import config
import matplotlib.patches as patches
from matplotlib import pyplot as plt


class SumSquaredErrorLoss():
    def __init__(self):
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def __call__(self, p, a):
        obj_ij = bbox_attr(a, 4)            # Indicator variable, 1 if grid I, bbox J contains object
        noobj_ij = 1 - obj_ij               # Opposite, 1's if grid I, bbox J contains NO object
        obj_i = obj_ij[:, :, 0:1]           # 1 if grid I has any object at all

        # XY position losses
        x_pos = bbox_attr(p, 0) - bbox_attr(a, 0)
        y_pos = bbox_attr(p, 1) - bbox_attr(a, 1)
        pos_losses = x_pos ** 2 + y_pos ** 2

        # Bbox dimension losses
        width = torch.sqrt(bbox_attr(p, 2)) - torch.sqrt(bbox_attr(a, 2))
        height = torch.sqrt(bbox_attr(p, 3)) - torch.sqrt(bbox_attr(a, 3))
        dim_losses = width ** 2 + height ** 2

        # Confidence losses
        confidence_losses = (bbox_attr(p, 4) - obj_ij) ** 2

        # Classification losses
        class_losses = (p[:, :, 5*config.B:] - a[:, :, 5*config.B:]) ** 2

        return torch.sum(obj_ij * (self.lambda_coord * (pos_losses + dim_losses) + confidence_losses)) \
               + torch.sum(obj_i * class_losses) \
               + torch.sum(noobj_ij * self.lambda_noobj * confidence_losses)


def load_classes():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_classes(new_dict)
    return new_dict


def save_classes(obj):
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

    return data[:, :, i:5*config.B:5]


def plot_boxes(data, labels):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S

    fig, ax = plt.subplots()
    plt.imshow(data.permute(1, 2, 0))
    for i in range(labels.size(dim=0)):
        for j in range(labels.size(dim=1)):
            for k in range(config.B):
                bbox = labels[i, j, 5 * k:5 * (k + 1)]
                bbox_center = (
                    bbox[0] + j * grid_size_x - bbox[2] / 2,
                    bbox[1] + i * grid_size_y - bbox[3] / 2
                )
                rect = patches.Rectangle(
                    bbox_center,
                    bbox[2],
                    bbox[3],
                    facecolor='none',
                    linewidth=1,
                    edgecolor='orange'
                )
                ax.add_patch(rect)
    plt.show()
