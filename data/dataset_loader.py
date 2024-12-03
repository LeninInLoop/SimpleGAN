from model import (
    DataLoader,
    torchvision,
    os,
    random_split
)

class DatasetDownloader:
    def __init__(self, dataset_path, transformers, train_size):
        self.transformers = transformers
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.download = not os.path.exists(dataset_path)

    def get_full_dataset(self):
        full_dataset = torchvision.datasets.MNIST(
            root=self.dataset_path,
            train=True,
            download=self.download,
            transform=self.transformers
        )
        test_dataset = torchvision.datasets.MNIST(
            root=self.dataset_path,
            train=False,
            download=self.download,
            transform=self.transformers
        )

        train_size = int(self.train_size * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size]
        )

        return train_dataset, val_dataset, test_dataset


class DatasetLoader(DatasetDownloader):
    def __init__(self, dataset_path, batch_size, num_workers, transformers, train_size):
        super().__init__(dataset_path, transformers, train_size)
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformers = transformers
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_full_dataset()

    def get_train_dataset(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def get_val_dataset(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def get_test_dataset(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
