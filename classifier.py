import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Nin(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1e-10):
        super(Nin, self).__init__()

        n = (in_channels + out_channels) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = torch.nn.Parameter(torch.zeros((in_channels, out_channels), dtype=torch.float32
                                                ).uniform_(-limit, limit))
        self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bcl, co->bol', x, self.W) + self.b

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super(AttentionBlock, self).__init__()
        self.scale = 0.1
        self.Q = Nin(ch, ch)
        self.K = Nin(ch, ch)
        self.V = Nin(ch, ch)
        self.nin = Nin(ch, ch, scale=0.)

    def forward(self, x):
        h = nn.functional.group_norm(x, num_groups=16)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)

        w = F.softmax(torch.einsum('bcl, bcl->bl', q, k).unsqueeze(-1), dim=1)
        h = v * w.permute(0, 2, 1)
        h = self.nin(h)
        return x + h

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.2):
        super(ResNetBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.convolution1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.convolution2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if not (in_ch == out_ch):
            self.nin = Nin(in_ch, out_ch)

    def forward(self, x):
        h = self.convolution1(x)
        h = nn.functional.leaky_relu(nn.functional.group_norm(h, num_groups=16), 0.4)
        h = nn.functional.dropout(h, p=self.dropout_rate)
        h = self.convolution2(h)
        h = nn.functional.leaky_relu(nn.functional.group_norm(h, num_groups=16), 0.4)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)

        if not (x.shape[2] == h.shape[2]):
            x = x[:, :, :h.shape[2]]

        return x + h

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv(x)
        return h

class ConvolutionalAttention(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ConvolutionalAttention, self).__init__()
        self.output_layer = nn.Linear(256, num_classes)
        self.input_projection = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResNetBlock(embed_dim, embed_dim),
            ResNetBlock(embed_dim, embed_dim),
            ResNetBlock(embed_dim, 2 * embed_dim),
            Encoder(2 * embed_dim),
            ResNetBlock(2 * embed_dim, 2 * embed_dim),
            ResNetBlock(2 * embed_dim, 2 * embed_dim),
            ResNetBlock(2 * embed_dim, 4 * embed_dim),
            AttentionBlock(4 * embed_dim),
            ResNetBlock(4 * embed_dim, 4 * embed_dim),
            ResNetBlock(4 * embed_dim, 4 * embed_dim),
            ResNetBlock(4 * embed_dim, 4 * embed_dim),
        ])
        self.final_attention = AttentionBlock(4 * embed_dim)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)

        h = self.final_attention(x)
        w = F.softmax(h, dim=1)

        out = torch.sum(x * w, dim=1)
        out = F.softmax(self.output_layer(out), dim=1)
        return out
