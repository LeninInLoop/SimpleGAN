from model import torch

class Config:
    DATA_DIR = './data/mnist_dataset'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    TRAIN_SIZE = 0.8
    NUM_EPOCHS = 100
    LATENT_DIM = 128
    LEARNING_RATE = 3e-4
    LEAKY_RATE = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
