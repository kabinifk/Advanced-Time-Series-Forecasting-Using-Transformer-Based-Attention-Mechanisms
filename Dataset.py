import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=48, pred_len=24):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return x.unsqueeze(-1), y.unsqueeze(-1)
