import config
import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Conv 1
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),          # Conv 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),         # Conv 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(4):                              # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1)
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):                              # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1)
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        ]

        for _ in range(2):                              # Conv 6
            layers.append(nn.Conv2d(1024, 1024, kernel_size=3, padding=1))

        layers += [
            nn.Flatten(),
            nn.Linear(config.S * config.S * 1024, 4096),            # Linear 1
            nn.Linear(4096, config.S * config.S * self.depth)       # Linear 2
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )

    @staticmethod
    def loss_function(actual, predicted):
        pass


class Probe(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"\nProbe '{self.name}':")
        print(x.size())
        return x


if __name__ == '__main__':
    batch_size = 128
    test_model = YOLOv1()
    test_tensor = torch.rand((batch_size, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    result = test_model.forward(test_tensor)
    assert tuple(result.size()) == (128, config.S, config.S, test_model.depth)
