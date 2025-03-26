import argparse
import json
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import numpy as np

from tiny_edm.loss import EDMLoss
from tiny_edm.models import MLP, EDM
from tiny_edm.utils import generation, evaluation
from tiny_edm.datasets import get_dataset, dataparams
from tiny_edm.vis_utils import viz_generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--gen_batch_size", type=int, default=1000)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--sampler", type=str, default="edm", choices=["edm", "sde", "ode"]
    )
    parser.add_argument("--second_order", action="store_true")

    config = parser.parse_args()
    config.num_epochs = 0
    config_train = json.load(open(f"exps/{config.path_model}/config.json"))
    config.dataset = config_train["dataset"]
    config.hidden_size = config_train["hidden_size"]
    config.hidden_layers = config_train["hidden_layers"]
    config.embedding_size = config_train["embedding_size"]
    config.time_embedding = config_train["time_embedding"]
    config.input_embedding = config_train["input_embedding"]


    if config.device is not None:
        if torch.cuda.is_available():
            device = f"cuda:{config.device}"
    else:
        device = "cuda"
    if torch.cuda.is_available():
        device = torch.device(device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dataset_train, dim = get_dataset(config.dataset, n=config.gen_batch_size)
    if dim == 1:
        config.gen_batch_size *= 10000
    dataset_gen, dim = get_dataset(config.dataset, n=config.gen_batch_size)
    frame_dataset = dataset_gen[:][0].numpy()

    config.dim_out = dim
    config.sigma_min = dataparams["sigma_min"]
    config.sigma_max = dataparams["sigma_max"]
    config.sigma_data = dataparams["sigma_data"]
    config.rho = dataparams["rho"]
    config.P_mean = dataparams["P_mean"]
    config.P_std = dataparams["P_std"]
    config.S_min = 0.003
    config.S_max = float("inf")
    config.S_churn = 0
    config.S_noise = 1

    net = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
        dim_out=dim,
    ).to(device)

    model = EDM(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        sigma_data=config.sigma_data,
        net=net,
    ).to(device)
    outdir = f"exps/{config.path_model}"
    config.experiment_name = config.path_model
    model.load_state_dict(torch.load(os.path.join(outdir, "model.pth"), weights_only=True))  
    if torch.cuda.is_available():
        model = model.to(device)

    frames = []
    with torch.no_grad():
        final_frames = generation(config, model, device, -1, frames)

    # save the final frames with a name config
    path = f"exps/{config.experiment_name}/generations"
    os.makedirs(path, exist_ok=True)
    name = f"sampler_{config.sampler}_second_order_{config.second_order}_num_timesteps_{config.num_timesteps}_.npy"
    np.save(os.path.join(path, name), final_frames[-1])
    viz_generation(config, final_frames, frame_dataset, eval=True)
