import torch.nn as nn
import math
import collections


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = collections.OrderedDict()
        m['conv1'] = conv3x3(in_channels, channels, stride)
        m['bn1'] = nn.BatchNorm2d(channels)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(channels, channels)
        m['bn2'] = nn.BatchNorm2d(channels)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = collections.OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, channels, kernel_size=1, downsample=None)
        m['bn1'] = nn.BatchNorm2d(channels)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(channels, channels, kernel_size=1, stride=stride,
                               padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(channels)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(channels, channels * 4, kernel_size=1,
                               bias=False)
        m['bn3'] = nn.BatchNorm2d(channels * 4)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        super(ResNet, self).__init__()
        m = collections.OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1,
                               bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.group2 = nn.Sequential(
            collections.OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)
        return x

