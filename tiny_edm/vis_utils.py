import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tiny_edm.utils import timesteps
from tiny_edm.loss import EDMLoss


def scaling(config, t):
    # Scaling to plot. must be linear in t and equal to 1 at t=sigma_min and to sigma_max at t=sigma_max

    return 1 + (config.sigma_max - 1) / (config.sigma_max - config.sigma_min) * (
        t - config.sigma_min
    )


def create_gif_from_pngs(folder_path, filename, duration=100):
    outpath = os.path.join(
        folder_path.split("/")[0], folder_path.split("/")[1], filename
    )
    # List all files in the folder and filter out only PNG files
    images = sorted([file for file in os.listdir(folder_path) if file.endswith(".png")])

    # Full paths to images
    image_paths = [os.path.join(folder_path, img) for img in images]

    # Open each image and add to the list
    frames = [Image.open(img) for img in image_paths]

    # Save as GIF
    frames[0].save(
        outpath, save_all=True, append_images=frames[1:], duration=duration, loop=1
    )
    print(f"GIF saved at {outpath}")


def viz_generation_1D(config, final_frames, frame_dataset, eval=False):
    res = 200
    final_frames = np.stack(final_frames)
    time_steps = timesteps(
        config.num_timesteps, config.sigma_min, config.sigma_max, config.rho
    )
    time_steps = time_steps.cpu().numpy()
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    img = np.zeros((final_frames.shape[0], res))
    for i, f in enumerate(final_frames):
        f = f / scaling(config, time_steps[i])
        hist, _ = np.histogram(f, bins=res, range=(-6, 6))
        img[i] = hist / np.sum(hist)
    plt.imshow(
        img, aspect="auto", extent=[-6, 6, 0, final_frames.shape[0]], cmap="Blues"
    )
    plt.xlabel("x")
    plt.ylabel("timesteps")
    plt.title("samples")
    plt.subplot(1, 2, 2)
    hist, bins = np.histogram(frame_dataset[:, 0], bins=res, range=(-6, 6))

    plt.bar(bins[:-1], hist / np.sum(hist), width=(bins[1] - bins[0]), label="data")
    hist, bins = np.histogram(final_frames[-1], bins=res, range=(-6, 6))
    plt.bar(
        bins[:-1],
        hist / np.sum(hist),
        width=(bins[1] - bins[0],),
        label="generated",
        alpha=0.5,
    )
    plt.legend()
    plt.xlabel("x")

    if eval:
        plt.savefig(f"exps/{config.experiment_name}/generation_eval.png")
    else:
        plt.savefig(f"exps/{config.experiment_name}/generation.png")
    np.save(f"exps/{config.experiment_name}/final_frames.npy", final_frames)


def viz_generation_2D(config, final_frames, frame_dataset):
    dir = f"exps/{config.experiment_name}"
    outdir = f"exps/{config.experiment_name}/generation"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    times_steps = timesteps(
        config.num_timesteps, config.sigma_min, config.sigma_max, config.rho
    ).cpu().numpy()
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(reversed(final_frames)):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        if i>0:
            frame_dataset = frame_dataset + np.random.randn(*frame_dataset.shape)*np.sqrt(times_steps[-i-1]**2-times_steps[-i]**2)
        plt.scatter(frame_dataset[:, 0], frame_dataset[:, 1], alpha=0.5)
        plt.xlim(xmin*(1+times_steps[-i-1]), xmax*(1+times_steps[-i-1]))
        plt.ylim(ymin*(1+times_steps[-i-1]), ymax*(1+times_steps[-i-1]))
        plt.savefig(f"{outdir}/{len(final_frames)-i-1:04}.png")
        plt.close()
    create_gif_from_pngs(outdir, "generation.gif")
    np.save(f"{dir}/final_frames.npy", final_frames)


def viz_generation(config, final_frames, frame_dataset, eval=False):
    if config.dim_out == 1:
        viz_generation_1D(config, final_frames, frame_dataset, eval)
    elif config.dim_out == 2:
        viz_generation_2D(config, final_frames, frame_dataset)
    else:
        raise Exception("Unsupported dimensionality for visualization")


def viz_training_1D(config, frames):
    res = 500
    frames = np.stack(frames)
    plt.figure(figsize=(10, 10))
    img = np.zeros((frames.shape[0] + 1, res))
    for i, f in enumerate(frames):
        hist, _ = np.histogram(f, bins=res, range=(-6, 6))
        img[i] = hist

    plt.imshow(img, aspect="auto", extent=[-6, 6, 0, frames.shape[0]], cmap="Blues")
    plt.xlabel("x")
    plt.ylabel("Epochs")
    plt.title("samples")
    plt.savefig(f"exps/{config.experiment_name}/training.png")
    np.save(f"exps/{config.experiment_name}/frames.npy", frames)


def viz_training_2D(config, frames):
    frames = np.stack(frames)
    outdir = f"exps/{config.experiment_name}"
    imgdir = f"{outdir}/training"
    if os.path.exists(imgdir):
        shutil.rmtree(imgdir)
    os.makedirs(imgdir, exist_ok=True)

    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()
    create_gif_from_pngs(imgdir, "training.gif")
    np.save(f"{outdir}/frames.npy", frames)


def viz_training(config, frames):
    if config.dim_out == 1:
        viz_training_1D(config, frames)
    elif config.dim_out == 2:
        viz_training_2D(config, frames)
    else:
        raise Exception("Unsupported dimensionality for visualization")


def viz_loss(config, losses, losses_eval, dataset, losses_eval_allt, loss_fn):
    outdir = f"exps/{config.experiment_name}"
    plt.figure(figsize=(10, 5))
    # delete the extrema quantiles to have a better visualization
    plt.plot(
        np.array(range(len(losses))) * (len(dataset) // config.train_batch_size),
        losses,
        label="Train",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{outdir}/loss.png")
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.array(range(len(losses_eval))) * config.save_images_step,
        losses_eval,
        label="Eval",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{outdir}/loss_eval.png")
    plt.figure(figsize=(10, 5))
    times_steps = timesteps(
        config.num_timesteps, config.sigma_min, config.sigma_max, config.rho
    )[:-1]
    plt.plot(times_steps, losses_eval_allt)
    plt.xlabel("sigma")
    plt.ylabel("Eval Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Loss at different timesteps")
    plt.savefig(f"{outdir}/loss_eval_t.png")
    plt.figure(figsize=(10, 5))
    plt.plot(times_steps)
    plt.xlabel("i")
    plt.ylabel("t_i")
    plt.title("Timesteps")  
    plt.savefig(f"{outdir}/timesteps.png")
    plt.figure(figsize=(10, 5))
    plt.plot(times_steps, loss_fn.weight(times_steps))
    plt.xlabel("sigma")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("weight")
    plt.title("Weight of the loss")
    plt.savefig(f"{outdir}/weight.png")
