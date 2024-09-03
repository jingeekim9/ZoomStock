"""
The PyTorch implementation of DTML.
"""
import torch
from torch import nn
from torch.nn import functional as func
from torch.nn.modules.activation import MultiheadAttention


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, mlp_size=2048, nhead=8, dropout=0.1, activation="relu", norm=True):
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_size, nhead, dropout=dropout)

        self.linear1 = nn.Linear(hidden_size, mlp_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_size, hidden_size)

        self.norm = norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super().__setstate__(state)

    def forward(self, src, mask):
        src2, attn_map = self.self_attn(src, src, src, key_padding_mask=mask)
        src = src + self.dropout1(src2)
        if self.norm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.norm:
            src = self.norm2(src)
        return src, attn_map


class AttentionLayers(nn.Module):
    def __init__(self, hidden_size, dropout, heads=8, factor=4, layers=1, norm=True):
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
        out = x
        attn_maps = []
        for layer in self.layers:
            out, attn_map = layer(out, ~mask)
            attn_maps.append(attn_map)
        return out, attn_maps


class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.attention = attention

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.attention:
            attn = torch.bmm(out.transpose(0, 1), out[-1].unsqueeze(-1)).squeeze(2)
            attn = torch.softmax(attn, dim=1)
            print("ATTENTION")
            print((attn.t().unsqueeze(-1) * out).shape)
            out = (attn.t().unsqueeze(-1) * out).sum(dim=0)
            print(out.shape)
            print("ATTENTION SHAPE")
            
        else:
            attn = None
            out = out[-1]
        return out, attn
    
class DFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=2)
        x = abs(x)
        return x
    
class CNN_Block(nn.Module):
    def __init__(self, in_features, out_features=None, batch_norm_size=20, kernel_size=5, stride=1, dilation=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, dilation=dilation, padding = padding, bias=True)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class GlobalFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 6, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, spatial_size=None):
        x = x.to(torch.float32)
        x = torch.fft.rfft(x)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x)
        return x
 
class CircularConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 6, 2, dtype=torch.float32) * 0.02)
        
    def forward(self, x):
        # Get the shapes of the input tensors
        N, C, L_x = x.shape
        M = len(self.complex_weight)
        L = L_x + M - 1  # Length of the result

        # Pad sequences with zeros
        x_padded = torch.cat((x, torch.zeros(N, C, L - L_x, dtype=x.dtype).cuda()), dim=2)
        h_padded = torch.cat((self.complex_weight, torch.zeros(M - 1, 3, 2, dtype=self.complex_weight.dtype).cuda()))

        # Initialize result
        result = torch.zeros(N, C, L, dtype=x.dtype).cuda()

        # Perform circular convolution
        for n in range(L):
            for m in range(M):
                result[:, :, n] += 1
                # result[:, :, n] += torch.sum(x_padded[:, :, (n - m) % L] * h_padded[m], dim=1) 
        
        return result
    
class ZoomBlock(nn.Module):
    def __init__(self, in_features):
        super(self.__class__, self).__init__()
        self.cnn = CNN_Block(in_features, in_features)
        self.cnn_week = CNN_Block(in_features, in_features)
        self.dft = DFT()
        # self.global_filter = CNN_Block(in_features, in_features, kernel_size=in_features, padding=in_features)
        self.global_filter = GlobalFilter(in_features)
        # self.global_filter = CircularConvolution(in_features)
        self.max_pool = nn.AvgPool2d((11, 1), stride=1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, X_history, X_week):
        # print("DAILY: ", X_history.shape)
        # print("WEEKLY: ", X_week.shape)
        a = self.cnn(X_history)
        b = self.dft(X_history)
        c = self.global_filter(X_week)
        d = self.cnn_week(X_week)
        out = torch.cat((a, b, c, d), 2)
        # print("CONCAT: ", out.shape)
        out = self.max_pool(out)
        out = self.dropout(out).sum(axis=1)
        return out
    

class DTML(nn.Module):
    def __init__(self, in_features, num_stocks, input_hidden_size=36, hidden_size=64, beta=1, dropout=0.10, attn_heads=4,
                 attn_layers=1, mlp_factor=4, norm=True, window=50):
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
        s1 = (self.linear2.weight ** 2).sum()
        s2 = (self.linear2.bias ** 2).sum()
        return s1 + s2

    def forward(self, x, mask, with_attn=False):
        # print("INPUT: ", x.shape)
        out = x.transpose(2,3)
        # print("1st OUT: ", out.shape)
        # out = self.linear3(x)
        out = torch.tanh(out)
        out = out.view(-1, out.size(2), out.size(3))
        back = int(self.window/5)
        # out = self.zoom_block(out, out[:, :, ::2])
        out = self.zoom_block(out[:, :, -back:], out[:, :, ::5])
        # out, time_attn = self.lstm(out.transpose(0, 1)
        out = out.view(x.size(0), x.size(1), -1)

        # print("Zoom reshape: ", out.shape)
        out = self.linear1(out)
        if self.norm:
            out = self.layer_norm(out)
        out_index = out[:, -1, :].unsqueeze(1)
        out_stocks = out[:, :-1, :]
        out = out_stocks + self.beta * out_index

        out = out.transpose(0, 1)
        # print("BEFORE ATTENTION: ", out.shape)
        out, stock_attn = self.attention(out, mask)
        out = out.transpose(0, 1)
        out = self.dropout(torch.tanh(out))
        out = self.linear2(out).squeeze(-1)

        if with_attn:
            out = out, time_attn, stock_attn
        return out
