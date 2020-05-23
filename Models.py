import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.layer_convolution = nn.Sequential(nn.Conv2d(3, 10, 3, 1),  # 10*10*10
                                               nn.BatchNorm2d(10),
                                               nn.PReLU(),
                                               nn.MaxPool2d(2, 2),  # 5*5*10
                                               nn.Conv2d(10, 16, 3, 1),  # 3*3*16
                                               nn.BatchNorm2d(16),
                                               nn.PReLU(),
                                               nn.Conv2d(16, 32, 3, 1),  # 1*1*32
                                               nn.BatchNorm2d(32),
                                               nn.PReLU())
        self.layer_confidence = nn.Conv2d(32, 1, 1, 1)  # 置信度层输出
        self.layer_offsets = nn.Conv2d(32, 4, 1, 1)  # 偏移率层输出

    def forward(self, x):
        layer = self.layer_convolution(x)
        confidence_output = torch.sigmoid(self.layer_confidence(layer))
        offsets_output = self.layer_offsets(layer)

        return confidence_output, offsets_output


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.layer_convolution = nn.Sequential(nn.Conv2d(3, 28, 3, 1),  # 22*22*28
                                               nn.BatchNorm2d(28),
                                               nn.PReLU(),
                                               nn.MaxPool2d(3, 2, padding=1),  # 11*11*28
                                               nn.Conv2d(28, 48, 3, 1),  # 9*9*48
                                               nn.BatchNorm2d(48),
                                               nn.PReLU(),
                                               nn.MaxPool2d(3, 2),  # 4*4*48
                                               nn.Conv2d(48, 64, 2, 1),  # 3*3*64
                                               nn.BatchNorm2d(64),
                                               nn.PReLU())
        self.layer_MLP = nn.Linear(3*3*64, 128)
        self.layer_confidence = nn.Linear(128, 1)
        self.layer_offsets = nn.Linear(128, 4)
        self.prelu = nn.PReLU()

    def forward(self, x):
        layer = self.layer_convolution(x)
        layer = torch.reshape(layer, (-1, 64*3*3))
        layer = self.layer_MLP(layer)
        layer = self.prelu(layer)
        confidence_output = torch.sigmoid(self.layer_confidence(layer))
        offset_output = self.layer_offsets(layer)

        return confidence_output, offset_output

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.layer_convolution = nn.Sequential(nn.Conv2d(3, 32, 3, 1),  # 46*46*32
                                               nn.BatchNorm2d(32),
                                               nn.PReLU(),
                                               nn.MaxPool2d(3, 2, ceil_mode=True),  # 23*23*32
                                               nn.Conv2d(32, 64, 3, 1),  # 21*21*64
                                               nn.BatchNorm2d(64),
                                               nn.PReLU(),
                                               nn.MaxPool2d(3, 2),  # 10*10*64
                                               nn.Conv2d(64, 64, 3, 1),  # 8*8*64
                                               nn.BatchNorm2d(64),
                                               nn.PReLU(),
                                               nn.MaxPool2d(2, 2),  # 4*4*64
                                               nn.Conv2d(64, 128, 2, 1),  # 3*3*128
                                               nn.BatchNorm2d(128),
                                               nn.PReLU())
        self.layer_MLP = nn.Linear(128*3*3, 256)
        self.layer_confidence = nn.Linear(256, 1)
        self.layer_offsets = nn.Linear(256, 4)
        self.layer_landmarks = nn.Linear(256, 10)
        self.prelu = nn.PReLU()

    def forward(self, x):
        layer = self.layer_convolution(x)
        layer = torch.reshape(layer, (-1, 128*3*3))
        layer = self.layer_MLP(layer)
        layer = self.prelu(layer)
        confidence_output = torch.sigmoid(self.layer_confidence(layer))
        offsets = self.layer_offsets(layer)
        landmarks = self.layer_landmarks(layer)

        return confidence_output, offsets, landmarks


if __name__ == "__main__":
    a = torch.randn((1, 3, 1006, 1600))

    b = torch.randn((1, 3, 500, 600))

    x = PNet()(a)
    y = PNet()(b)
    # t = torch.reshape(a, (-1, 3*24*24))
    # f = a.view(a.shape[0], -1)
    z = torch.cat((x[0], y[0]), dim=1)
    print(x[0].shape)
    print(y[0].shape)
