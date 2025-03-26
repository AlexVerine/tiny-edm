import torch
import numpy as np
from tqdm import tqdm


def timesteps(num_steps, sigma_min, sigma_max, rho):
    step_indices = torch.arange(num_steps)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps


def edm_sde_step(net, x, t, tnext, i, S_churn, S_min, S_max, S_noise, num_steps, second_order=True):
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t <= S_max else 0
    t_hat = net.round_sigma(t + gamma * t)
    x_hat = x + (t_hat**2 - t**2).sqrt() * S_noise * torch.randn_like(x)

    denoised = net(x_hat, t_hat.repeat(x_hat.shape[0], 1))
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (tnext - t_hat) * d_cur

    if i < num_steps - 1 and second_order:
        denoised = net(x_next, tnext.repeat(x_next.shape[0], 1))
        d_prime = (x_next - denoised) / tnext
        x_next = x_hat + (tnext - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def sde_step(net, x, t, tnext):
    score = (x - net(x, t.repeat(x.shape[0], 1))) / t**2
    x_next = x - (t**2 - tnext**2) * score
    x_next = x_next + torch.sqrt(t**2 - tnext**2) * torch.randn_like(x_next)
    return x_next


def ode_step(net, x, t, tnext, i, num_steps, second_order=True):
    denoised = net(x, t.repeat(x.shape[0], 1))
    d_cur = (x - denoised) / t
    x_next = x + (tnext - t) * d_cur

    if i < num_steps - 1 and second_order:
        denoised = net(x_next, tnext.repeat(x_next.shape[0], 1))
        d_prime = (x_next - denoised) / tnext
        x_next = x + (tnext - t) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def edm_sampler(
    net,
    latents,
    config,
    return_latents=False,
):
    num_steps = config.num_timesteps
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max
    rho = config.rho
    S_churn = config.S_churn
    S_min = config.S_min
    S_max = config.S_max
    S_noise = config.S_noise

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    t_steps = timesteps(num_steps, sigma_min, sigma_max, rho).to(latents.device)
    t_steps = net.round_sigma(t_steps)  

    # Main sampling loop.
    x_next = latents * t_steps[0]

    if return_latents:
        latents = [x_next.cpu().numpy()]

    for i, (t_cur, t_next) in enumerate(
        tqdm(zip(t_steps[:-1], t_steps[1:]), leave=False, desc="Generation")
    ):  # 0, ..., N-1
        x_cur = x_next

        if config.sampler == "edm":
            x_next = edm_sde_step(
                net, x_cur, t_cur, t_next, i, S_churn, S_min, S_max, S_noise, num_steps, config.second_order
            )
        elif config.sampler == "sde":
            x_next = sde_step(net, x_cur, t_cur, t_next)

        elif config.sampler == "ode":
            x_next = ode_step(
                net, x_cur, t_cur, t_next, i, num_steps, config.second_order
            )
        if return_latents:
            latents.append(x_next.cpu().numpy())

    if return_latents:
        return latents
    else:
        return x_next.cpu().numpy()


def generation(config, model, device, epoch, frames):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(config.gen_batch_size, config.dim_out, device=device)
        if epoch == config.num_epochs - 1:
            final_frames = edm_sampler(model, sample, config, return_latents=True)
            return final_frames
        else:
            frames.append(edm_sampler(model, sample, config))
            return None


def evaluation(config, model, loss_fn, device, epoch, dataloader_eval, losses_eval):
    model.eval()
    with torch.no_grad():
        times_eval = timesteps(
            config.num_timesteps, config.sigma_min, config.sigma_max, config.rho
        )[:-1]
        loss_eval_allt = []
        for i, t in tqdm(enumerate(times_eval), desc="Evaluation", leave=False):
            t = torch.tensor([t] * config.eval_batch_size, device=device).view(-1, 1)
            loss_eval = 0
            for step, batch in enumerate(dataloader_eval):
                batch = batch[0].to(device)
                loss_eval += loss_fn.loss(model, batch, t).mean().item() * len(batch)
            loss_eval_allt.append(loss_eval / len(dataloader_eval.dataset))
        loss_eval_allt = np.array(loss_eval_allt)
        losses_eval.append(loss_eval_allt.mean())
    if epoch == config.num_epochs - 1:
        return loss_eval_allt
    else:
        return None
