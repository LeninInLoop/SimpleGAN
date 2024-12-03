from data import DatasetLoader, Transformers
from model import nn, Generator, Discriminator, optim, torch, torchvision, SummaryWriter
from config import Config

dataset = DatasetLoader(
    dataset_path=Config.DATA_DIR,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    transformers=Transformers.get_mnist_transform(),
    train_size=Config.TRAIN_SIZE,
)

generator = Generator(Config.LATENT_DIM).to(Config.DEVICE)
discriminator = Discriminator().to(Config.DEVICE)
fixed_random_noise = torch.randn(Config.BATCH_SIZE, Config.LATENT_DIM).to(Config.DEVICE)

# Define loss function and optimizer
criterion = nn.BCELoss()

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=Config.LEARNING_RATE
    )

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=Config.LEARNING_RATE
)

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(Config.NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataset.get_train_dataset()):
        real = real.view(-1, 28 * 28 * 1).to(Config.DEVICE)
        batch_size = real.shape[0]

        ### Train Discriminator: Max log( D(real) ) + log( 1 - D( G(z) ), Z is just a random Noise.
        noise = torch.randn(batch_size, Config.LATENT_DIM).to(Config.DEVICE)
        fake = generator(noise)

        discriminator_real = discriminator(real).view(-1)
        lossD_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

        discriminator_fake = discriminator(fake.detach()).view(-1)
        lossD_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))

        lossD = (lossD_real + lossD_fake) / 2
        discriminator.zero_grad()
        lossD.backward()
        optimizer_D.step()

        ### Train Generator: Min log( 1 - D( G(z) ) <-> max log( D(G(z)) )
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        lossG.backward()
        optimizer_G.step()

        if batch_idx == 0:
            print(f"Epoch: {epoch}/{Config.NUM_EPOCHS}, LossD: {lossD:.4f}, LossG: {lossG:.4f}")

            with torch.no_grad():
                fake = generator(fixed_random_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
