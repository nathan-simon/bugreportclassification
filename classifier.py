import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Nin(nn.Module):
    """
    1x1 convolution class, used as part of the multi-head attention mechanism.
    Inspired from https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Denoising_Diffusion_Probabilistic_Models/unet.py
    Adapted for 1D data.

    :param in_channels: number of input dimensions
    :param out_channels: number of output dimensions
    :return: scaled output tensor
    """
    def __init__(self, in_channels, out_channels, scale=1.0):
        super(Nin, self).__init__()
        n = (in_channels + out_channels) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = torch.nn.Parameter(torch.zeros((in_channels, out_channels), dtype=torch.float32
                                                 ).uniform_(-limit, limit))
        self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bcl, co->bol', x, self.W) + self.b


class AttentionBlock(nn.Module):
    """
    Multihead Attention Mechanism, each attention output concatenated and then finally linearly transformed to correct dimensions.
    Enables the CNN to focus on different aspects of the "Bag of Words" provided as input, enabling richer understanding of data.

    :param num_channels: number of channels in the input tensor
    :param num_heads: number of heads to use in attention mechanism
    :return: final output (concat of all attention outputs)
    """
    def __init__(self, num_channels, num_heads):
        super(AttentionBlock, self).__init__()
        assert num_channels % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = num_channels
        self.num_heads = num_heads
        self.head_dim = num_channels // num_heads

        self.query = Nin(num_channels, num_channels)
        self.key = Nin(num_channels, num_channels)
        self.value = Nin(num_channels, num_channels)

        self.out = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        batch_size, num_channels, seq_length  = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, num_channels)

        output = self.out(attn_output).permute(0, 2, 1)
        return output

class ResNetBlock(nn.Module):
    """
    Residual Block of CNN. Performs convolutions on data, as well as dropout (random deactivations of neurons)
    Using Leaky ReLU as activation function, as well as group normalisation.

    :param in_channels: number of input dimensions
    :param out_channels: number of output dimensions
    :param dropout_rate: probability of deactivation of neuron

    :return: convolved tensor, ready to be passed to next layer
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResNetBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.convolution1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.convolution2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if not (in_channels == out_channels):
            self.nin = Nin(in_channels, out_channels)

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
    """
    Encoding mechanism, convolves input with a stride of 2 to downsize the input vector, capturing abstract features of the data.

    :param in_channels: number of input dimensions
    :return: tensor with half the sequence length
    """
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv(x)
        return h

class ConvolutionalAttention(nn.Module):
    """
    Convolutional Attention Module.
    Main module that acts as a classifier. 3 residual layers between each subsidiary layer.
    Final output involves softmax, sum and fully connected output layer to compute logits.

    :param: num_channels: number of input dimensions
    :param: num_classes: number of classes to predict (in this case 2 since binary classification)
    :return: tensor with logits
    """
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
