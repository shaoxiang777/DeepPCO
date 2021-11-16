import torch.nn as nn
from models.sub_network import TranslationSubNet, RotationSubNet


class DeepPCO(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_net = TranslationSubNet()
        self.r_net = RotationSubNet()

    def forward(self, x):
        t_t, t_r = self.t_net(x)
        r_t, r_r = self.r_net(x)
        return t_t, t_r, r_t, r_r
