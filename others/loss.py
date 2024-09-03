from torch import nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, output, target):
        return self.relu(1 - (output * (2 * target - 1)))
