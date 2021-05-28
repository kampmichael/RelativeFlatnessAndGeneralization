from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, output_dim, input_dim = None):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        init.xavier_normal_(self.conv2.weight.data)
        init.zeros_(self.conv2.bias.data)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(120, 84)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(84, output_dim)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def feature_layer(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.view(-1, 84, 1)

    def classify(self, x):
        x = self.fc3(x)
        return x

