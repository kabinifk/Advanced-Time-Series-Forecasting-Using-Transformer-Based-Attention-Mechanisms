import torch.nn as nn
import math
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(500, d_model)
        pos = torch.arange(0, 500).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, 2, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.fc = nn.Linear(d_model, 24)

    def forward(self, x):
        x = self.pos(self.in_proj(x))
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 24)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).unsqueeze(-1)
