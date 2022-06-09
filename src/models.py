from torchvision.models import vgg16_bn as VGG16
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        vgg16 = VGG16(pretrained=True)
        self.features = vgg16.features

    def forward(self, x):
        """
        torch.Size([1, 128, 128, 128])
        torch.Size([1, 256, 64, 64])
        torch.Size([1, 512, 32, 32])
        torch.Size([1, 512, 16, 16])
        """
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")
        self.conv21 = nn.Conv2d(1024, 128, 1, 1)
        self.bn21 = nn.BatchNorm2d(128)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(128, 128, 3, 1, padding="same")
        self.bn22 = nn.BatchNorm2d(128)
        self.relu22 = nn.ReLU()

        self.conv31 = nn.Conv2d(384, 64, 1, 1)
        self.bn31 = nn.BatchNorm2d(64)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(64, 64, 3, 1, padding="same")
        self.bn32 = nn.BatchNorm2d(64)
        self.relu32 = nn.ReLU()

        self.conv41 = nn.Conv2d(192, 32, 1, 1)
        self.bn41 = nn.BatchNorm2d(32)
        self.relu41 = nn.ReLU()
        self.conv42 = nn.Conv2d(32, 32, 3, 1, padding="same")
        self.bn42 = nn.BatchNorm2d(32)
        self.relu42 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 32, 3, 1, padding="same")
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

    def forward(self, inputs):
        """
            output: torch.Size([1, 32, 128, 128])
        """
        out4, out3, out2, out1 = inputs
        x = self.upsample(out1)
        x = torch.cat([out2, x], 1)
        x = self.relu21(self.bn21(self.conv21(x)))
        x = self.relu22(self.bn22(self.conv22(x)))

        x = self.upsample(x)
        x = torch.cat([out3, x], 1)
        x = self.relu31(self.bn31(self.conv31(x)))
        x = self.relu32(self.bn32(self.conv32(x)))

        x = self.upsample(x)
        x = torch.cat([out4, x], 1)
        x = self.relu41(self.bn41(self.conv41(x)))
        x = self.relu42(self.bn42(self.conv42(x)))

        x = self.relu5(self.bn5(self.conv5(x)))
        return x


class Output(nn.Module):
    def __init__(self, scope=512):
        super(Output, self).__init__()
        self.scope = 512
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        """
            output: [torch.Size([1, 1, 128, 128]),
                     torch.Size([1, 5, 128, 128])]
        """
        scores = self.sigmoid1(self.conv1(x))
        locations = self.sigmoid2(self.conv2(x)) * self.scope
        angles = self.sigmoid3(self.conv3(x)) * math.pi
        geo = torch.cat((locations, angles), 1)
        return scores, geo


class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.backbone = Backbone()
        self.merge = Merge()
        self.output = Output()

    def forward(self, x):
        return self.output(self.merge(self.backbone(x)))


if __name__ == "__main__":
    east = EAST()
    img = torch.randn(1, 3, 512, 512)

    for result in east(img):
        print(result.shape)
