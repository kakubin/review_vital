import math
import torch
from torch import nn


class PositionalEncoder(torch.nn.Module):
    def __init__(self, seq_len=512, d_model=768):
        super().__init__()

        self.dropout = nn.Dropout()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    tensor = torch.rand([24, 512, 768])
    net = PositionalEncoder(max_seq_len=512, d_model=768)
    print(net(tensor).shape)
