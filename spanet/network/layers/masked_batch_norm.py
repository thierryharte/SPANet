from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import init

# Pytorch Masked BatchNorm
# Based on this implementation
# https://gist.github.com/yangkky/364413426ec798589463a3a88be24219


class MaskedBatchNorm1D(nn.Module):

    __constants__ = ["num_features", "eps", "momentum", "affine", "track_running_stats"]

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(MaskedBatchNorm1D, self).__init__()

        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        # Register affine transform learnable parameters
        if affine:
            self.weight = nn.Parameter(torch.Tensor(1, 1, num_features))
            self.bias = nn.Parameter(torch.Tensor(1, 1, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Register moving average storable parameters
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, 1, num_features))
            self.register_buffer('running_var', torch.ones(1, 1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Calculate the masked mean and variance
        timesteps, batch_size, feature_dim = features.shape
        mask = mask.view(timesteps, batch_size)

        if mask is not None:
            masked_features = features[mask]
        else:
            masked_features = features.view(timesteps * batch_size, feature_dim)

        # Compute masked image statistics
        current_mean = masked_features.mean(0).view(1, 1, feature_dim).detach()
        current_var = masked_features.var(0).view(1, 1, feature_dim).detach()

        # Update running statistics
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var

            self.num_batches_tracked += 1

        # Apply running statistics transform
        if self.track_running_stats and not self.training:
            normed_images = (features - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed_images = (features - current_mean) / (torch.sqrt(current_var + self.eps))

        # Apply affine transform from learned parameters
        if self.affine:
            normed_images = normed_images * self.weight + self.bias

        return normed_images * mask.unsqueeze(2)
