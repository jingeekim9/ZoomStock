import torch
from torch import nn


class DARNN(nn.Module):
    def __init__(self, num_variables, window, encoder_size=64, decoder_size=64):
        super().__init__()
        self.encoder = Encoder(num_variables, window, encoder_size)
        self.decoder = Decoder(encoder_size, decoder_size)

    def forward(self, x, target):
        # x is of size (batch_size x num_variables (n) x window (T)).
        y = x[:, target, :]
        x_tilde, h = self.encoder(x)
        return self.decoder(x_tilde, h, y)


class Encoder(nn.Module):
    def __init__(self, num_variables, window, hidden_size=64):
        super().__init__()
        self.window = window
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_variables, hidden_size)
        self.linear_v = nn.Linear(window, 1, bias=False)
        self.linear_w = nn.Linear(hidden_size * 2, window)
        self.linear_u = nn.Linear(window, window)

    def run_attention(self, h_last, c_last, u_out):
        out = self.linear_w(torch.cat((h_last, c_last), dim=1))
        out = self.linear_v(torch.tanh(out.unsqueeze(1) + u_out)).squeeze(2)
        return torch.softmax(out, dim=1)

    def forward(self, x):
        # x is of size (batch_size x num_variables (n) x window (T)).
        assert x.size(2) == self.window
        batch_size = x.size(0)
        u_out = self.linear_u(x)
        h_last = torch.zeros((batch_size, self.hidden_size), device=x.device)
        c_last = torch.zeros((batch_size, self.hidden_size), device=x.device)
        x_tilde_list, h_list = [], []
        for i in range(self.window):
            alpha = self.run_attention(h_last, c_last, u_out)
            x_tilde = alpha * x[:, :, i]
            _, (h, c) = self.lstm(x_tilde.unsqueeze(0), (h_last.unsqueeze(0), c_last.unsqueeze(0)))
            h_last, c_last = h[-1], c[-1]
            h_list.append(h_last)
            x_tilde_list.append(x_tilde)
        return torch.stack(x_tilde_list), torch.stack(h_list)


class Decoder(nn.Module):
    def __init__(self, encoder_size, decoder_size):
        super().__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.lstm = nn.LSTM(1, decoder_size)
        self.linear_v = nn.Linear(decoder_size, 1, bias=False)
        self.linear_w = nn.Linear(decoder_size * 2, encoder_size)
        self.linear_u = nn.Linear(encoder_size, encoder_size)
        self.linear_p = nn.Linear(decoder_size + 1, 1)
        self.linear_y = nn.Linear(decoder_size * 2, 1)

    def run_attention(self, h_last, c_last, u_out):
        w_out = self.linear_w(torch.cat((h_last, c_last), dim=1))  # batch_size x encoder_size (m)
        v_out = self.linear_v(torch.tanh(w_out + u_out)).squeeze(2)
        return torch.softmax(v_out, dim=0)

    def forward(self, x, h_encoder, y):
        # x.size = window (T) x batch_size x num_variables (n)
        # h_encoder.size = window (T) x batch_size x encoder_size (m)
        # y.size = batch_size x window (T)
        window = x.size(0)
        batch_size = x.size(1)
        h_last = torch.zeros((batch_size, self.decoder_size), device=x.device)
        c_last = torch.zeros((batch_size, self.decoder_size), device=x.device)
        u_out = self.linear_u(h_encoder)  # window (T) x batch_size
        for i in range(1, window):
            y_prev = y[:, i - 1]
            beta = self.run_attention(h_last, c_last, u_out)
            context = torch.sum(beta.unsqueeze(2) * h_encoder, dim=0)  # batch_size x encoder_size (m)
            y_tilde = self.linear_p(torch.cat((y_prev.unsqueeze(1), context), dim=1))
            _, (h, c) = self.lstm(y_tilde.unsqueeze(0), (h_last.unsqueeze(0), c_last.unsqueeze(0)))
            h_last, c_last = h[-1], c[-1]
        return self.linear_y(torch.cat((h_last, c_last), dim=1)).squeeze(1)
