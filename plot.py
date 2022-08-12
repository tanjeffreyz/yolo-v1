import torch
import config
import utils
from data import YoloPascalVocDataset
from models import YOLOv1
from torch.utils.data import DataLoader


WEIGHTS_PATH = 'models/yolo_v1/08_11_2022/18_11_08/weights/final'


def show_test_images():
    classes = utils.load_class_array()
    test_set = YoloPascalVocDataset('test')
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE)
    model = YOLOv1()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    with torch.no_grad():
        for image, labels in test_loader:
            print(image.size(), labels.size())
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(image[i, :, :], predictions[i, :, :], classes, threshold=0.1)


if __name__ == '__main__':
    show_test_images()
