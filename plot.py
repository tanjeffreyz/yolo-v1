import torch
import config
import utils
from data import YoloPascalVocDataset
from models import *
from torch.utils.data import DataLoader


WEIGHTS_PATH = 'models/yolo_v1/08_17_2022/19_05_22/weights/final'


def show_test_images():
    classes = utils.load_class_array()

    dataset = 'test'
    clean_set = YoloPascalVocDataset(dataset)
    test_set = YoloPascalVocDataset(dataset, augment=True)
    clean_loader = DataLoader(clean_set, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE)

    model = YOLOv1ResNet()
    model.eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH))

    with torch.no_grad():
        for (image, labels), (clean, _) in zip(test_loader, clean_loader):
            print(image.size(), '->', labels.size())
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(
                    clean[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    threshold=0.2
                )


if __name__ == '__main__':
    show_test_images()
