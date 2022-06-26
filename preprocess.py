import os
import json
import pickle
import config
import utils
import numpy as np
import torchvision.transforms as T
from torchvision.datasets.voc import VOCDetection


def preprocess(dataset, folder):
    path = os.path.join(config.DATA_ROOT, folder)
    if not os.path.exists(path):
        os.makedirs(path)
    for i, pair in enumerate(dataset):
        data, label = pair
        


if __name__ == '__main__':
    train_set = VOCDetection(
        root=config.DATA_ROOT,
        year='2007',
        image_set='train',
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])
    )

    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()

    data, label = next(iter(train_set))
    plt.imshow(data.permute(1, 2, 0))

    for xmin, xmax, ymin, ymax in utils.get_bounding_boxes(label):
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin
        )
        ax.add_patch(rect)
    plt.show()
    exit()

    test_set = VOCDetection(
        root=config.DATA_ROOT,
        year='2007',
        image_set='val',
        transform=T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])
    )

    preprocess(train_set, 'train')
    preprocess(test_set, 'test')
