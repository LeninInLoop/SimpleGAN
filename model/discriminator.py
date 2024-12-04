from model import nn
from config import Config

class Discriminator(nn.Module):
    """
    Discriminator network that classifies an input image as real or fake.

    The Discriminator is a CNN-based model that downscales the image and outputs a probability
    (real or fake) using a Sigmoid activation.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(Config.LEAKY_RATE),
            nn.Linear(256, 128),
            nn.LeakyReLU(Config.LEAKY_RATE),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability (real or fake)
        )

    def forward(self, img):
        """
        Forward pass through the Discriminator.

        Args:
            img (Tensor): Input image tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Tensor: Output probability for each image in the batch (real or fake).
        """
        return self.model(img)
