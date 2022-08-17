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

    def __init__(self, folder, transform=None):
        assert folder in {'train', 'test'}
        self.transform = transform
        self.files = glob.glob(os.path.join('data', folder, '*'))

    def __getitem__(self, i):
        d, t = torch.load(self.files[i])
        if self.transform is not None:
            d = self.transform(d)
        return d, t

    def __len__(self):
        return len(self.files)


def preprocess(folder):
    classes = utils.load_class_dict()
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
        target = os.path.join(output_dir, str(i))
        if not os.path.exists(target):
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
                    classes[name] = YoloPascalVocDataset.index
                    YoloPascalVocDataset.index += 1
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
                        (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coordinate relative to grid square
                        (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coordinate relative to grid square
                        (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width
                        (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height
                        1.0                                                     # Confidence
                    )
                    bbox_start = 5 * bbox_index
                    bbox_end = 5 * (bbox_index + 1)
                    ground_truth[row, col, bbox_start:bbox_end] = torch.tensor(bbox_truth)
                    boxes[key] = bbox_index + 1

                # Insert class one-hot encoding into ground truth
                one_hot = torch.zeros(config.C)
                one_hot[class_index] = 1.0
                ground_truth[row, col, -config.C:] = one_hot

            # Save preprocessed data
            torch.save((data, ground_truth), target)
    utils.save_class_dict(classes)


if __name__ == '__main__':
    # Preprocess data
    # preprocess('train')
    # preprocess('test')

    # Display data
    obj_classes = utils.load_class_array()
    train_set = YoloPascalVocDataset('train')
    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        print(label[:, :, -config.C:])
        utils.plot_boxes(data, label, obj_classes)
    print('num_negatives', negative_labels)
    print('dist', smallest, largest)
