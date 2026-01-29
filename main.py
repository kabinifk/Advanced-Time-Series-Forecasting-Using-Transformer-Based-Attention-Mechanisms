import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_generator import generate_series
from dataset import TimeSeriesDataset
from model import TransformerModel, LSTMModel
from utils import metrics, plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

series = generate_series()
train, test = train_test_split(series, test_size=0.2, shuffle=False)

train_ds = TimeSeriesDataset(train)
test_ds = TimeSeriesDataset(test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

def train(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(5):
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print("Loss:", total/len(train_loader))

def evaluate(model):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu())
            trues.append(y)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

# Transformer
t_model = TransformerModel().to(device)
train(t_model)
pred_t, true = evaluate(t_model)

# LSTM
l_model = LSTMModel().to(device)
train(l_model)
pred_l, _ = evaluate(l_model)

print("Transformer:", metrics(pred_t, true))
print("LSTM:", metrics(pred_l, true))

plot(true, pred_t, pred_l)
