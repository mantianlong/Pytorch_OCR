# write for fun, can u finish alone
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # input_channel = 1, output_channel = 6, kernel_size = 5
        # input_size = (32, 32)  out_size = (28, 28) (input + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        # input_size = (28, 28)  out_size = (14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # input_size = (14, 14)  out_size = (10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)
        # (10, 10) -> (5, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    lenet = LeNet()
    x = torch.Tensor(100, 1, 32, 32)
    y = lenet(x)
    print(y.size())
