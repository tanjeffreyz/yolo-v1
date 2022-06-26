from torch.utils.data import DataLoader


class YoloPascalVocDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)



