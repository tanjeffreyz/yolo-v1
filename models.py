import config
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)              # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)


class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = 5 * config.B + config.C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, config.S, config.S, self.depth)
        )


###########################
#       From Scratch      #
###########################
class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        layers = [
            # Probe(0, forward=lambda x: print('#' * 5 + ' Start ' + '#' * 5)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),                   # Conv 1
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv1', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),                           # Conv 2
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv2', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),                                     # Conv 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv3', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(4):                                                          # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv4', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):                                                          # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv5', forward=probe_dist),
        ]

        for _ in range(2):                                                          # Conv 6
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        # layers.append(Probe('conv6', forward=probe_dist))

        layers += [
            nn.Flatten(),
            nn.Linear(config.S * config.S * 1024, 4096),                            # Linear 1
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('linear1', forward=probe_dist),
            nn.Linear(4096, config.S * config.S * self.depth),                      # Linear 2
            # Probe('linear2', forward=probe_dist),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )


#############################
#       Helper Modules      #
#############################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


class Probe(nn.Module):
    names = set()

    def __init__(self, name, forward=None):
        super().__init__()

        assert name not in self.names, f"Probe named '{name}' already exists"
        self.name = name
        self.names.add(name)
        self.forward = self.probe_func_factory(probe_size if forward is None else forward)

    def probe_func_factory(self, func):
        def f(x):
            print(f"\nProbe '{self.name}':")
            func(x)
            return x
        return f


def probe_size(x):
    print(x.size())


def probe_mean(x):
    print(torch.mean(x).item())


def probe_dist(x):
    print(torch.min(x).item(), '|', torch.median(x).item(), '|', torch.max(x).item())
