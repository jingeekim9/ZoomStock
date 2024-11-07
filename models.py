"""
ZoomStock (BigData 2024)

Authors:
    - JinGee Kim (jingeekim9@snu.ac.kr)
    - Yong-chan Park (wjdakf3948@snu.ac.kr)
    - Jaemin Hong (jmhong0120@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

Affiliation:
    - Data Mining Lab., Seoul National University

File: models.py
     - The PyTorch implementation of ZoomStock.

Version: 1.0.0
"""

import torch
from torch import nn
from torch.nn import functional as func
from torch.nn.modules.activation import MultiheadAttention


# Utility function to return the appropriate activation function
# Input:
#   - activation: str, the name of the activation function ('relu' or 'gelu')
# Output:
#   - The specified activation function from torch.nn.functional
def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# Transformer encoder layer with multi-head self-attention and a feed-forward network
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, mlp_size=2048, nhead=8, dropout=0.1, activation="relu", norm=True):
        """
        Initializes the TransformerEncoderLayer.
        Inputs:
        - hidden_size: int, the dimensionality of the input and output
        - mlp_size: int, the dimensionality of the feed-forward network
        - nhead: int, the number of attention heads
        - dropout: float, dropout rate applied after attention and feed-forward layers
        - activation: str, activation function to use in the feed-forward network
        - norm: bool, whether to apply layer normalization
        """
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_size, nhead, dropout=dropout)

        # Feed-forward network with two linear layers
        self.linear1 = nn.Linear(hidden_size, mlp_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_size, hidden_size)

        # Normalization layers
        self.norm = norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function (ReLU or GELU)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, mask):
        """
        Forward pass for the encoder layer.
        Inputs:
        - src: Tensor, shape [sequence_length, batch_size, hidden_size], input sequence
        - mask: Tensor, shape [batch_size, sequence_length], key padding mask for attention

        Outputs:
        - src: Tensor, shape [sequence_length, batch_size, hidden_size], transformed sequence
        - attn_map: Tensor, attention map from the self-attention layer
        """
        src2, attn_map = self.self_attn(src, src, src, key_padding_mask=mask)
        src = src + self.dropout1(src2)
        if self.norm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.norm:
            src = self.norm2(src)
        return src, attn_map


# Stack of Transformer encoder layers
class AttentionLayers(nn.Module):
    def __init__(self, hidden_size, dropout, heads=8, factor=4, layers=1, norm=True):
        """
        Initializes the AttentionLayers module.
        Inputs:
        - hidden_size: int, dimensionality of the input and output of each layer
        - dropout: float, dropout rate applied after attention and feed-forward layers
        - heads: int, the number of attention heads per layer
        - factor: int, scaling factor for the hidden size of the feed-forward layers
        - layers: int, number of Transformer encoder layers to stack
        - norm: bool, whether to apply layer normalization
        """
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                mlp_size=hidden_size * factor,
                nhead=heads,
                dropout=dropout,
                norm=norm)
            for _ in range(layers))

    def forward(self, x, mask):
        """
        Forward pass for the stack of attention layers.
        Inputs:
        - x: Tensor, shape [sequence_length, batch_size, hidden_size], input sequence
        - mask: Tensor, shape [batch_size, sequence_length], key padding mask for attention

        Outputs:
        - out: Tensor, shape [sequence_length, batch_size, hidden_size], transformed sequence
        - attn_maps: List of attention maps for each layer
        """
        out = x
        attn_maps = []
        for layer in self.layers:
            out, attn_map = layer(out, ~mask)
            attn_maps.append(attn_map)
        return out, attn_maps


# LSTM with optional attention mechanism
class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention=True):
        """
        Initializes the AttnLSTM module.
        Inputs:
        - input_size: int, dimensionality of input features
        - hidden_size: int, dimensionality of LSTM hidden state
        - num_layers: int, number of LSTM layers
        - attention: bool, whether to apply attention mechanism on LSTM output
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.attention = attention

    def forward(self, x):
        """
        Forward pass for the LSTM with optional attention.
        Inputs:
        - x: Tensor, shape [sequence_length, batch_size, input_size], input sequence

        Outputs:
        - out: Tensor, shape [batch_size, hidden_size], LSTM output with optional attention
        - attn: Tensor or None, attention weights if attention is applied, otherwise None
        """
        out, _ = self.lstm(x)
        if self.attention:
            attn = torch.bmm(out.transpose(0, 1), out[-1].unsqueeze(-1)).squeeze(2)
            attn = torch.softmax(attn, dim=1)
            out = (attn.t().unsqueeze(-1) * out).sum(dim=0)
        else:
            attn = None
            out = out[-1]
        return out, attn
    

# Discrete Fourier Transform (DFT) layer to transform inputs to frequency domain
class DFT(nn.Module):
    def forward(self, x):
        """
        Forward pass for the DFT layer.
        Input:
        - x: Tensor, shape [batch_size, channels, sequence_length], input sequence

        Output:
        - x: Tensor, transformed to the frequency domain (magnitude only)
        """
        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=2)
        x = abs(x)  # Magnitude of complex numbers
        return x
    

# Convolutional block with batch normalization and ReLU activation
class CNN_Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=5, stride=1, dilation=1, padding=2):
        """
        Initializes the CNN_Block.
        Inputs:
        - in_features: int, number of input channels
        - out_features: int, number of output channels
        - kernel_size, stride, dilation, padding: int, convolutional parameters
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass for the CNN block.
        Input:
        - x: Tensor, shape [batch_size, in_features, sequence_length], input sequence

        Output:
        - x: Tensor, shape [batch_size, out_features, sequence_length], transformed sequence
        """
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

# Filter that transforms inputs in the frequency domain and applies complex weights
class GlobalFilter(nn.Module):
    def __init__(self, dim):
        """
        Initializes the GlobalFilter.
        Input:
        - dim: int, number of channels in the input sequence
        """
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 6, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        """
        Forward pass for the GlobalFilter.
        Input:
        - x: Tensor, shape [batch_size, channels, sequence_length], input sequence in the frequency domain

        Output:
        - x: Tensor, shape [batch_size, channels, sequence_length], transformed in the frequency domain
        """
        x = x.to(torch.float32)
        x = torch.fft.rfft(x)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x)
        return x
 

# Experimental circular convolution layer
class CircularConvolution(nn.Module):
    def forward(self, x):
        """
        Forward pass for CircularConvolution (currently incomplete).
        Input:
        - x: Tensor, shape [batch_size, channels, sequence_length], input sequence

        Output:
        - result: Tensor, transformed sequence using circular convolution (not fully implemented)
        """
        N, C, L_x = x.shape
        M = len(self.complex_weight)
        L = L_x + M - 1

        x_padded = torch.cat((x, torch.zeros(N, C, L - L_x, dtype=x.dtype).cuda()), dim=2)
        h_padded = torch.cat((self.complex_weight, torch.zeros(M - 1, 3, 2, dtype=self.complex_weight.dtype).cuda()))

        result = torch.zeros(N, C, L, dtype=x.dtype).cuda()

        for n in range(L):
            for m in range(M):
                result[:, :, n] += 1
        return result
    

# Combines CNN, DFT, and filtering blocks to process multiple temporal scales
class ZoomBlock(nn.Module):
    def __init__(self, in_features):
        """
        Initializes the ZoomBlock.
        Input:
        - in_features: int, number of input channels
        """
        super().__init__()
        self.cnn = CNN_Block(in_features, in_features)
        self.cnn_week = CNN_Block(in_features, in_features)
        self.dft = DFT()
        self.global_filter = GlobalFilter(in_features)
        self.max_pool = nn.AvgPool2d((11, 1), stride=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X_history, X_week):
        """
        Forward pass for the ZoomBlock.
        Inputs:
        - X_history: Tensor, shape [batch_size, channels, sequence_length], historical sequence data
        - X_week: Tensor, shape [batch_size, channels, sequence_length], weekly sequence data

        Output:
        - out: Tensor, shape [batch_size, features], aggregated multi-scale output
        """
        a = self.cnn(X_history)
        b = self.dft(X_history)
        c = self.global_filter(X_week)
        d = self.cnn_week(X_week)
        out = torch.cat((a, b, c, d), 2)
        out = self.max_pool(out)
        out = self.dropout(out).sum(axis=1)
        return out
    

# Main ZoomStock model combining multiple components for temporal analysis
class ZoomStock(nn.Module):
    def __init__(self, in_features, num_stocks, input_hidden_size=36, hidden_size=64, beta=1, dropout=0.10, attn_heads=4,
                 attn_layers=1, mlp_factor=4, norm=True, window=50):
        """
        Initializes the ZoomStock model.
        Inputs:
        - in_features: int, number of input channels
        - num_stocks: int, number of stocks (used for the layer norm shape)
        - input_hidden_size: int, hidden size for input linear transformation
        - hidden_size: int, hidden size for LSTM and attention layers
        - beta: float, scaling factor for attention outputs
        - dropout: float, dropout rate
        - attn_heads: int, number of attention heads in the attention layers
        - attn_layers: int, number of attention layers
        - mlp_factor: int, scaling factor for the hidden size of the feed-forward layers
        - norm: bool, whether to apply layer normalization
        - window: int, size of the time window for analysis
        """
        super().__init__()
        self.beta = beta
        self.norm = norm
        self.window = window
        self.lstm = AttnLSTM(hidden_size, hidden_size, num_layers=1, attention=True)
        self.linear3 = nn.Linear(in_features, hidden_size)
        self.linear1 = nn.Linear(input_hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionLayers(
            hidden_size, dropout, attn_heads, mlp_factor, attn_layers, norm=norm)
        self.layer_norm = nn.LayerNorm((num_stocks + 1, hidden_size))
        self.zoom_block = ZoomBlock(in_features)

    def l2_loss(self):
        """
        Computes L2 regularization loss for linear layer weights and biases.
        Output:
        - Regularization loss: float
        """
        s1 = (self.linear2.weight ** 2).sum()
        s2 = (self.linear2.bias ** 2).sum()
        return s1 + s2

    def forward(self, x, mask, with_attn=False):
        """
        Forward pass for ZoomStock.
        Inputs:
        - x: Tensor, shape [batch_size, num_stocks, sequence_length, in_features], input stock data
        - mask: Tensor, shape [batch_size, num_stocks, sequence_length], mask for attention
        - with_attn: bool, whether to return attention maps

        Outputs:
        - out: Tensor, shape [batch_size, num_stocks], final stock prediction
        - stock_attn (optional): attention map from the attention layers if with_attn is True
        """
        out = x.transpose(2,3)
        out = torch.tanh(out)
        out = out.view(-1, out.size(2), out.size(3))
        back = int(self.window/5)
        out = self.zoom_block(out[:, :, -back:], out[:, :, ::5])
        out = out.view(x.size(0), x.size(1), -1)

        out = self.linear1(out)
        if self.norm:
            out = self.layer_norm(out)
        out_index = out[:, -1, :].unsqueeze(1)
        out_stocks = out[:, :-1, :]
        out = out_stocks + self.beta * out_index

        out = out.transpose(0, 1)
        out, stock_attn = self.attention(out, mask)
        out = out.transpose(0, 1)
        out = self.dropout(torch.tanh(out))
        out = self.linear2(out).squeeze(-1)

        if with_attn:
            return out, stock_attn
        return out
