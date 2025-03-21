import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Nin(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):  # increased scale from 1e-10 to 1.0
        super(Nin, self).__init__()
        n = (in_channels + out_channels) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = torch.nn.Parameter(torch.zeros((in_channels, out_channels), dtype=torch.float32
                                                 ).uniform_(-limit, limit))
        self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bcl, co->bol', x, self.W) + self.b


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = Nin(embed_dim, embed_dim)
        self.W_k = Nin(embed_dim, embed_dim)
        self.W_v = Nin(embed_dim, embed_dim)

        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, embed_dim, seq_length  = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        output = self.W_o(attn_output).permute(0, 2, 1)
        return output

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
    def __init__(self, num_channels, num_classes):
        super(ConvolutionalAttention, self).__init__()
        self.output_layer = nn.Linear(4 * num_channels, num_classes)
        self.input_projection = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResNetBlock(num_channels, num_channels),
            ResNetBlock(num_channels, num_channels),
            ResNetBlock(num_channels, 2 * num_channels),
            Encoder(2 * num_channels),
            ResNetBlock(2 * num_channels, 2 * num_channels),
            ResNetBlock(2 * num_channels, 2 * num_channels),
            ResNetBlock(2 * num_channels, 2 * num_channels),
            AttentionBlock(2 * num_channels, 8),
            ResNetBlock(2 * num_channels, 2 * num_channels),
            ResNetBlock(2 * num_channels, 4 * num_channels),
            ResNetBlock(4 * num_channels, 4 * num_channels),
        ])
        self.final_attention = AttentionBlock(4 * num_channels, 16)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)

        h = self.final_attention(x)
        w = F.softmax(h, dim=2)
        out = torch.sum(x * w, dim=2)

        out = self.output_layer(out).squeeze(1)
        return out
