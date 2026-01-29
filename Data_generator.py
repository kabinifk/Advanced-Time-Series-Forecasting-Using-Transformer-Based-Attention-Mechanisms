import numpy as np

def generate_series(n=2000):
    np.random.seed(42)
    t = np.arange(n)

    trend = 0.05 * t
    season = 10 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 3, n)

    series = trend + season + noise
    return (series - series.mean()) / series.std()
