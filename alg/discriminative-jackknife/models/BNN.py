import torch
import torch.nn as nn


class BNN(nn.Module):
    def __init__(self, *layers):
        super(BNN, self).__init__()

        self.layers, self.params = [], nn.ParameterList()
        for layer in layers:
            self.layers.append(layer)
            self.params.extend([*layer.parameters()])   # register module parameters

    def forward(self, x, mode):
        if mode == 'forward':
            net_kl = 0
            for layer in self.layers:
                x, layer_kl = layer.forward(x, mode)
                net_kl += layer_kl
            return x, net_kl
        else:
            for layer in self.layers:
                x = layer.forward(x, mode)
            return x

    def Forward(self, x, y, n_samples, type):

        assert type in {'Gaussian', 'Softmax'}, 'Likelihood type not found'

        # Sample N samples and average
        total_kl, total_likelh = 0., 0.
        for _ in range(n_samples):
            out, kl = self.forward(x, mode='forward')

            # Gaussian output (with unit var)
            # lklh = torch.log(torch.exp(-(y - out) ** 2 / 2e-2) / math.sqrt(2e-2 * math.pi)).sum()

            if type == 'Gaussian':
                lklh = (-.5 * (y - out) ** 2).sum()
            else:   # softmax
                lklh = torch.log(out.gather(1, y)).sum()

            total_kl += kl
            total_likelh += lklh

        return total_kl / n_samples, total_likelh / n_samples

    @staticmethod
    def loss_fn(kl, lklh, n_batch):
        return (kl / n_batch - lklh).mean()
