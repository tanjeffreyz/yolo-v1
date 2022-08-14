import os
import glob
import torch
import config
import utils
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset


class YoloPascalVocDataset(Dataset):
    index = 0

    def __init__(self, folder):
        assert folder in {'train', 'test'}
        target = os.path.join('data', folder)
        self.classes = utils.load_class_dict()
        if not os.path.exists(target):
            self.preprocess(folder)
            utils.save_class_dict(self.classes)
        self.files = glob.glob(os.path.join(target, '*'))

    def preprocess(self, folder):
        dataset = VOCDetection(
            root=config.DATA_PATH,
            year='2007',
            image_set=('train' if folder == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
        )

        output_dir = os.path.join(config.DATA_PATH, folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, data_pair in enumerate(tqdm(dataset, desc=f'Preprocessing {folder}')):
            data, label = data_pair
            grid_size_x = data.size(dim=2) / config.S       # Images in PyTorch have size (channels, height, width)
            grid_size_y = data.size(dim=1) / config.S

            # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
            boxes = {}
            depth = 5 * config.B + config.C         # 5 numbers per bbox, then one-hot encoding of label
            ground_truth = torch.zeros((config.S, config.S, depth))
            for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
                name, coords = bbox_pair
                if name not in self.classes:
                    self.classes[name] = YoloPascalVocDataset.index
                    YoloPascalVocDataset.index += 1
                class_index = self.classes[name]
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
                        (mid_x - col * grid_size_x) / grid_size_x,      # X coordinate relative to grid square
                        (mid_y - row * grid_size_y) / grid_size_y,      # Y coordinate relative to grid square
                        (x_max - x_min) / config.IMAGE_SIZE[0],         # Width
                        (y_max - y_min) / config.IMAGE_SIZE[1],         # Height
                        1.0                                             # Confidence
                    )
                    bbox_start = 5 * bbox_index
                    bbox_end = 5 * (bbox_index + 1)
                    ground_truth[row, col, bbox_start:bbox_end] = torch.tensor(bbox_truth)
                    boxes[key] = bbox_index + 1

                # Insert class one-hot encoding into ground truth
                one_hot = torch.zeros(config.C)
                one_hot[class_index] = 1
                ground_truth[row, col, -config.C:] = one_hot

            # Save preprocessed data
            torch.save((data, ground_truth), os.path.join(output_dir, str(i)))

    def __getitem__(self, i):
        return torch.load(self.files[i])

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    classes = utils.load_class_array()
    train_set = YoloPascalVocDataset('train')
    num_negatives = 0
    for data, label in train_set:
        num_negatives += torch.sum(label < 0).item()
        # utils.plot_boxes(data, label, classes)
    print(num_negatives)
