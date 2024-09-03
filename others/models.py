from torch import nn


class AR(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x.view(x.size(0), x.size(1), -1)).squeeze(-1)

    @staticmethod
    def l2_loss():
        return 0


class MLP(nn.Module):
    def __init__(self, in_features, num_units=16, num_layers=2):
        super().__init__()
        assert num_layers >= 2
        layers = [nn.Linear(in_features, num_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_units, num_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_units, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.view(x.size(0), x.size(1), -1)).squeeze(-1)
