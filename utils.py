import numpy as np
import matplotlib.pyplot as plt

def metrics(pred, true):
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return mae, rmse

def plot(true, pred1, pred2):
    plt.figure(figsize=(12,4))
    plt.plot(true[:200], label="True")
    plt.plot(pred1[:200], label="Transformer")
    plt.plot(pred2[:200], label="LSTM")
    plt.legend()
    plt.show()
