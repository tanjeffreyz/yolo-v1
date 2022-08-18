import torch
import config
import utils
from data import YoloPascalVocDataset
from models import *
from torch.utils.data import DataLoader


WEIGHTS_PATH = 'models/yolo_v1/08_18_2022/13_16_20/weights/final'


def show_test_images():
    classes = utils.load_class_array()

    dataset = YoloPascalVocDataset('test', normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=8)

    model = YOLOv1ResNet()
    model.eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH))

    with torch.no_grad():
        for image, labels, original in loader:
            print(image.size(), '->', labels.size())
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(
                    original[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    threshold=0.1
                )


if __name__ == '__main__':
    show_test_images()
