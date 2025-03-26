import torch


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def sample_sigma(self, y):
        rnd_normal = torch.randn([y.shape[0], 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigma

    def loss(self, net, y, sigma):
        weight = self.weight(sigma)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        # loss = (D_yn - y) ** 2
        return loss

    def __call__(self, net, y):
        sigma = self.sample_sigma(y)
        loss = self.loss(net, y, sigma)
        return loss
