import json
import os
import config
import matplotlib.patches as patches
from matplotlib import pyplot as plt


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
