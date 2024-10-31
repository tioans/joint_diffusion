from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self, n_channels=9):
        super(Model, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(256, 120)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(84, 10)

        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # self.dropout_1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool1d(kernel_size=2)
        # self.dropout_2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3)
        self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool1d(kernel_size=2)
        # self.dropout_3 = nn.Dropout(dropout_rate)
        
        # Formula: Output = [(W-K+2P)/S]+1
        # num_in_features = ((9 - 3 + 2*0) / 1) + 1
        num_in_features = 9
        # print("Num in features: ", num_in_features * 16)
        self.fc1 = nn.Linear(in_features=int(16 * num_in_features), out_features=128)
        self.relu4 = nn.ReLU()
        # self.dropout_4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 84)
        self.relu5 = nn.ReLU()
        # self.dropout_5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(84, 2)


    def forward(self, x):
        # y = self.conv1(x)
        # y = self.relu1(y)
        # y = self.pool1(y)
        # y = self.conv2(y)
        # y = self.relu2(y)
        # y = self.pool2(y)
        # y = y.view(y.shape[0], -1)
        # y = self.fc1(y)
        # y = self.relu3(y)
        # y = self.fc2(y)
        # y = self.relu4(y)
        # y = self.fc3(y)
        # return y

        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        # y = self.dropout_1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        # y = self.pool2(y)
        # y = self.dropout_2(y)

        y = self.conv3(y)
        y = self.relu3(y)
        # y = self.pool3(y)
        # y = self.dropout_3(y)
        print("After conv3: ", y.shape)

        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu4(y)
        # y = self.dropout_4(y)
        y = self.fc2(y)
        y = self.relu5(y)
        # y = self.dropout_5(y)
        y = self.fc3(y)
        # y = torch.softmax(y, dim=1)
        return y

    ##### Part forward without last classification layer for the purpose of FID computing
    def part_forward(self, x):
        # y = self.conv1(x)
        # y = self.relu1(y)
        # y = self.pool1(y)
        # y = self.conv2(y)
        # y = self.relu2(y)
        # y = self.pool2(y)
        # y = y.view(y.shape[0], -1)
        # y = self.fc1(y)
        # y = self.relu3(y)
        # y = self.fc2(y)
        # return y

        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        # y = self.pool2(y)

        y = self.conv3(y)
        y = self.relu3(y)
        # y = self.pool3(y)

        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu4(y)
        y = self.fc2(y)

        return y


lenet = Model()