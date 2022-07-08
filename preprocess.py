import os
import pickle
import torch
import config
import utils
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection


index = 0
classes = utils.load_classes()


def preprocess(dataset, folder):
    global index

    output_dir = os.path.join(config.DATA_PATH, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, data_pair in enumerate(tqdm(dataset, desc=folder)):
        data, label = data_pair
        grid_size_x = data.size(dim=2) / config.S       # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        depth = 5 * config.B + config.C         # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            if name not in classes:
                classes[name] = index
                index += 1
            class_index = classes[name]
            x_min, x_max, y_min, y_max = coords
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2

            # Calculate the position of center of bounding box
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            # Insert bounding box into ground truth tensor
            key = (row, col)
            bbox_index = boxes.get(key, 0)
            if bbox_index < config.B:
                bbox_truth = (
                    mid_x - col * grid_size_x,  # X coordinate relative to grid square
                    mid_y - row * grid_size_y,  # Y coordinate relative to grid square
                    x_max - x_min,              # Width
                    y_max - y_min,              # Height
                    1.0                         # Confidence
                )
                for k, value in enumerate(bbox_truth):
                    ground_truth[row][col][5 * bbox_index + k] = value
                boxes[key] = bbox_index + 1

            # Insert class one-hot encoding into ground truth
            one_hot = torch.zeros(config.C)
            one_hot[class_index] = 1
            ground_truth[row, col, -config.C:] = one_hot
        torch.save(ground_truth, os.path.join(output_dir, str(i)))

        # TODO: SHOW THE IMAGE FOR TESTING PURPOSES
        # from matplotlib import pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots()
        # plt.imshow(data.permute(1, 2, 0))
        # for i in range(ground_truth.size(dim=0)):
        #     for j in range(ground_truth.size(dim=1)):
        #         for k in range(config.B):
        #             bbox_truth = ground_truth[i, j, 5*k:5*(k+1)]
        #             rect = patches.Rectangle(
        #                 (bbox_truth[0] + j * grid_size_x - bbox_truth[2] / 2, bbox_truth[1] + i * grid_size_y - bbox_truth[3] / 2),
        #                 bbox_truth[2],
        #                 bbox_truth[3],
        #                 facecolor='none',
        #                 linewidth=2,
        #                 edgecolor='orange'
        #             )
        #             ax.add_patch(rect)
        # plt.show()


if __name__ == '__main__':
    train_set = VOCDetection(
        root=config.DATA_PATH,
        year='2007',
        image_set='train',
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])
    )
    test_set = VOCDetection(
        root=config.DATA_PATH,
        year='2007',
        image_set='val',
        transform=T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])
    )

    preprocess(train_set, 'train')
    preprocess(test_set, 'test')
    utils.save_classes(classes)
