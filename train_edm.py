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
from tiny_edm.vis_utils import viz_loss, viz_training, viz_generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base", help="Name of the experiment folder.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dino",
        choices=["circle", "dino", "line", "moons", "gaussians", "1Dgaussians"],
        help="Dataset to use for training.",
    )
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=1000, help="Batch size for evaluation")
    parser.add_argument("--gen_batch_size", type=int, default=1000, help="Batch size for generation")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of steps for generation")
    parser.add_argument("--embedding_size", type=int, default=128, help="Size of the embeddings")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layers")
    parser.add_argument("--hidden_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument(
        "--time_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "zero"],
        help="Type of time embedding to use.",
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "identity"],
        help="Type of input embedding to use.",
    )
    parser.add_argument("--save_images_step", type=int, default=1, help="Save images every n epochs")
    parser.add_argument("--path_model", type=str, default=None, help="Path to model to continue training")
    parser.add_argument("--device", type=int, default=None, help="Device to use for training")
    parser.add_argument(
        "--sampler", type=str, default="edm", choices=["edm", "sde", "ode"], help="Sampler to use for generation"
    )
    parser.add_argument("--second_order", action="store_true", help="Use second order ODE/edm solver")

    config = parser.parse_args()

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

    dataset_train, dim = get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset_train, batch_size=config.train_batch_size, shuffle=True, drop_last=True
    )
    dataset_eval, dim = get_dataset(config.dataset, n=config.eval_batch_size)
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=True,
        drop_last=True,
    )
    if dim == 1:
        config.gen_batch_size *= 1000
    dataset_gen, dim = get_dataset(config.dataset, n=config.gen_batch_size)
    frame_dataset = dataset_gen[:][0].numpy()

    config.dim_out = dim
    config.sigma_min = dataparams["sigma_min"]
    config.sigma_max = dataparams["sigma_max"]
    config.sigma_data = dataparams["sigma_data"]
    config.rho = dataparams["rho"]
    config.P_mean = dataparams["P_mean"]
    config.P_std = dataparams["P_std"]
    config.S_min = 0
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

    if config.path_model is not None:
        model.load_state_dict(torch.load(os.path.join(config.path_model, "model.pth")))
        global_step = len(np.load(os.path.join(config.path_model, "loss.npy")))
        losses = list(np.load(os.path.join(config.path_model, "loss.npy")))
        losses_eval = list(np.load(os.path.join(config.path_model, "loss_eval.npy")))

        frames = list(np.load(os.path.join(config.path_model, "frames.npy")))
        frames_label = list(
            np.load(os.path.join(config.path_model, "frames_labels.npy"))
        )
    else:
        global_step = 0
        losses = []
        losses_eval = []
        frames = []
        frames_label = []

    
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    loss_fn = EDMLoss(
        P_mean=config.P_mean, P_std=config.P_std, sigma_data=config.sigma_data
    )
    global_step = 0
    frames = []
    losses = []
    print(f"Training model starting from step {global_step}...")

    pbar_epoch = trange(config.num_epochs, desc="Epoch")
    for epoch in pbar_epoch:
        model.train()
        pbar_batch = tqdm(dataloader, leave=False)
        for step, batch in enumerate(pbar_batch):
            batch = batch[0].to(device)
            loss = loss_fn(model, batch).mean()

            if global_step % 10 == 0:
                loss_mean = loss.detach().item()
                if global_step > 0:
                    losses.append(loss_mean / 10)
                    pbar_batch.set_postfix_str(f"Loss: {losses[-1]:.4f}")
            else:
                loss_mean += loss.detach().item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # if global_step == 10:
            #     losses_eval.append(losses[-1])
            global_step += 1

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            lists = generation(
                config,
                model,
                device,
                epoch,
                frames,
            )
            lists2 = evaluation(
                config,
                model,
                loss_fn,
                device,
                epoch,
                dataloader_eval,
                losses_eval,
            )
            if epoch == config.num_epochs - 1:
                final_frames = lists
                losses_eval_allt = lists2
        pbar_epoch.set_postfix_str(f"Loss Eval: {losses_eval[-1]:.4f}")

    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
    # save config using json
    with open(f"{outdir}/config.json", "w") as f:
        json.dump(vars(config), f)

    print("Plotting results...")
    viz_loss(config, losses, losses_eval, dataset_train, losses_eval_allt, loss_fn)
    viz_training(config, frames)
    viz_generation(config, final_frames, frame_dataset)

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", losses)
    np.save(f"{outdir}/loss_eval.npy", losses_eval)
