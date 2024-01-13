import numpy as np
from torch.utils.data import Dataset


data_root = '/home/intelligence/Robin/Dataset'
# data_root = 'D:\\Exercise\\Dataset_preprocess'


class base_loader(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)

