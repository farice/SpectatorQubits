import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(SimpleBody, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.fc1 = nn.Linear(input_shape[0], 16) if not self.noisy else NoisyLinear(input_shape[0], 16, sigma_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def feature_size(self):
        return self.fc1(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
