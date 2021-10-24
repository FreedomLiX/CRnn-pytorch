import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    """
    B W hidden_size* (1 or 2) ==》 B W num_class
    num_class
    """
    def __init__(self, in_channels, num_class, mid_channels=None):
        super(CTCHead, self).__init__()
        self.mid_channels = mid_channels
        self.num_class = num_class

        if mid_channels is None:
            self.fc = nn.Linear(in_channels, self.num_class)
        else:
            self.fc1 = nn.Linear(in_channels, self.mid_channels)
            self.fc2 = nn.Linear(self.mid_channels, self.num_class)

    def forward(self, x):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts