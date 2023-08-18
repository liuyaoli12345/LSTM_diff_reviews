import os
import pandas as pd
from math import ceil
from torch.utils.data import Dataset

class IMDBTrainingDataset(Dataset):

    def __init__(self, chunk_size=1000, transform=None, target_transform=None):
        self.chunk_size = chunk_size
        self.transform = transform
        self.target_transform = target_transform
        self.total_samples = self.get_total_samples()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self))

        chunks = pd.read_csv('vector_label.csv', skiprows=range(1, start_idx), nrows=self.chunk_size, header=None)
        vectors = chunks.iloc[:, 0].tolist()
        labels = chunks.iloc[:, 1].tolist()

        return vectors, labels

    def get_total_samples(self):
        num_rows = sum(1 for _ in pd.read_csv('vector_label.csv', chunksize=self.chunk_size, header=None))
        return num_rows * self.chunk_size
