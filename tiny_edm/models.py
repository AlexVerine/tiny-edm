import torch
from torch import nn
from tiny_edm.positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        dim_out=2,
        **kwargs
    ):
        super().__init__()
        self.dim_out = dim_out
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        if dim_out == 2:
            self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = (
            (
                len(self.time_mlp.layer)
                + len(self.input_mlp1.layer)
                + len(self.input_mlp2.layer)
            )
            if dim_out == 2
            else len(self.time_mlp.layer) + len(self.input_mlp1.layer)
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, dim_out))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        if self.dim_out == 2:
            x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[2])
        x = (
            torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
            if self.dim_out == 2
            else torch.cat((x1_emb, t_emb), dim=-1)
        )
        x = self.joint_mlp(x)
        return x


class EDM(nn.Module):
    def __init__(self, sigma_min, sigma_max, sigma_data, net):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.net = net

    def forward(self, x, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        Fx = self.net(c_in * x, c_noise)

        return c_out * Fx + c_skip * x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# class DDIM(nn.Module):
#     def __init__(self, C1, C2, M, net):
#         super().__init__()
#         self.C1 = C1
#         self.C2 = C2
#         self.M = M
#         u = torch.zeros(M + 1)
#         for j in range(M, 0, -1):  # M, ..., 1
#             u[j - 1] = (
#                 (u[j] ** 2 + 1)
#                 / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
#                 - 1
#             ).sqrt()
#         self.sigma_min = float(u[M - 1])
#         self.sigma_max = float(u[0])
#         self.net = net

#     def forward(self, x, sigma):
#         c_skip = 1
#         c_out = -sigma
#         c_in = 1 / (sigma**2 + 1).sqrt()
#         c_noise = self.M - 1 - self.round_sigma(sigma)
#         Fx = self.net(c_in * x, c_noise)
#         return c_out * Fx + c_skip * x

#     def round_sigma(self, sigma, return_index=False):
#         sigma = torch.as_tensor(sigma)
#         index = torch.cdist(
#             sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
#             self.u.reshape(1, -1, 1),
#         ).argmin(2)
#         result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
#         return result.reshape(sigma.shape).to(sigma.device)

#     def alpha_bar(self, j):
#         j = torch.as_tensor(j)
#         return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
