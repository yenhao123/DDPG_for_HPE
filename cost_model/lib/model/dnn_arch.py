import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, n_counters):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(n_counters, 2048)

        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)

        self.act_fn = nn.LeakyReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x