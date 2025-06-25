import torch
import numpy as np

class SpikingNeuralNetwork:
    def __init__(self):
        pass
    
    def poisson_encode(self, images, duration=100):
        """
        Convert static images (pixel 0–1) to Poisson spike trains.
        Args:
            images (Tensor): [batch, 1, 28, 28]
            duration (int): number of time steps
        Returns:
            spikes (Tensor): [batch, 1, 28, 28, time]
        """
        batch_size = images.shape[0]
        images = images.unsqueeze(-1)  # shape: [B, 1, 28, 28, 1]
        random_tensor = torch.rand((batch_size, 1, 28, 28, duration))
        spikes = (random_tensor < images).float()
        return spikes  # 0/1 spike tensor