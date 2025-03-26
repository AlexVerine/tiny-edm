import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


dataparams = {
    "sigma_min": 0.002,
    "sigma_max": 100,
    "sigma_data": 0.1,
    "rho": 7,
    "P_mean": -1.2,
    "P_std": 1.2,
}


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def gaussians_dataset(n=8000):
    centers_x = np.linspace(-4, 4, 5)
    centers_y = np.linspace(-4, 4, 5)
    x = np.random.randn(n) * 0.15
    y = np.random.randn(n) * 0.15
    centers = np.random.choice(centers_x, n), np.random.choice(centers_y, n)
    X = np.stack((x, y), axis=1)
    X[:, 0] += centers[0]
    X[:, 1] += centers[1]
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def oneDgaussians(n=8000):
    centers_x = np.linspace(-4, 4, 5)
    x = np.random.randn(n) * 0.15
    centers = np.random.choice(centers_x, n)
    X = x.reshape(-1, 1)
    X += centers.reshape(-1, 1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def oneDgaussiansUnbalanced(n=8000):
    centers_x = np.linspace(-4, 4, 5)
    x = np.random.randn(n) * 0.15
    centers = np.random.choice(centers_x, n, p=[0.3, 0.1, 0.2, 0.1, 0.3])
    X = x.reshape(-1, 1)
    X += centers.reshape(-1, 1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=10000):
    if name == "moons":
        return moons_dataset(n), 2
    elif name == "dino":
        return dino_dataset(n), 2
    elif name == "line":
        return line_dataset(n), 2
    elif name == "circle":
        return circle_dataset(n), 2
    elif name == "gaussians":
        return gaussians_dataset(n), 2
    elif name == "1Dgaussians":
        return oneDgaussians(n), 1
    elif name == "1DgaussiansUnbalanced":
        return oneDgaussiansUnbalanced(n), 1
    else:
        raise ValueError(f"Unknown dataset: {name}")
