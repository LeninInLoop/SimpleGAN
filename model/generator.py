from config import Config
from model import nn

class Generator(nn.Module):
    """
    Generator network that takes a latent vector (noise) and generates an image.

    Args:
        latent_dim (int): The size of the latent (input) vector.
    """

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Define the generator network structure
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(Config.LEAKY_RATE),
            nn.Linear(256, 128),
            nn.LeakyReLU(Config.LEAKY_RATE),
            nn.Linear(128, 28 * 28 * 1),
            nn.Tanh()  # Output image normalized to [-1, 1], we would normalize the mnist dataset too.
        )

    def forward(self, z):
        """
        Forward pass through the generator network.

        Args:
            z (Tensor): Latent vector of size (batch_size, latent_dim).

        Returns:
            Tensor: Generated image of size (batch_size, 1, 28, 28).
        """
        return self.model(z)
