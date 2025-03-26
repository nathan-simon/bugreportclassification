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
        h = F.leaky_relu(F.group_norm(h, num_groups=16), 0.4)
        h = F.dropout(h, p=self.dropout_rate)
        h = self.convolution2(h)
        h = F.leaky_relu(F.group_norm(h, num_groups=16), 0.4)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)

        if not (x.shape[2] == h.shape[2]):
            x = x[:, :, :h.shape[2]]

        return x + h

class ConvolutionalAttention(nn.Module):
    """
    Convolutional Attention Module.
    Main module that acts as a classifier. 3 residual layers between each subsidiary layer.
    Final output involves softmax, sum and fully connected output layer to compute logits.

    :param: vocab_size: dimension of vocabulary from tokenisation
    :param: num_channels: number of input dimensions
    :param: num_classes: number of classes to predict (in this case 2 since binary classification)
    :param: fc_dim: dimension of fully connected layer
    :return: tensor with logits
    """
    def __init__(self, vocab_size, num_channels, num_classes, fc_dim):
        super(ConvolutionalAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_channels)
        self.input_projection = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResNetBlock(num_channels, num_channels),
            ResNetBlock(num_channels, 2 * num_channels),
            ResNetBlock(2 * num_channels, 2 * num_channels)
        ])
        self.multihead_attention = nn.MultiheadAttention(embed_dim=2 * num_channels, num_heads=4, dropout=0.1)
        self.dropout = nn.Dropout(0.4)
        self.fully_connected = nn.Linear(2 * num_channels, fc_dim)
        self.output_layer = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        x = x.permute(2, 0, 1)
        h, _ = self.multihead_attention(x, x, x)
        h = h.permute(1, 2, 0)

        pooled_out = F.max_pool1d(h, kernel_size=h.shape[2]).squeeze(2)

        out = self.dropout(pooled_out)
        out = F.leaky_relu(self.fully_connected(out), 0.4)
        out = self.dropout(out)

        logits = self.output_layer(out)
        return logits.squeeze(1)
