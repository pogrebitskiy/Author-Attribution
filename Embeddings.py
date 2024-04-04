"""
This file contains the Embeddings class, which is used to load the embeddings from the .npy files

By: David Pogrebitskiy and Jacob Ostapenko
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Embeddings:
    def __init__(self, doc2vec_file, bert_file, labels_file):
        # Load the embeddings from the .npy files
        self.doc2vec = np.load(doc2vec_file)
        self.bert = np.load(bert_file)
        self.labels = np.load(labels_file)

    def get_doc2vec(self, as_torch=False):
        if as_torch:
            return torch.from_numpy(self.doc2vec)
        else:
            return self.doc2vec

    def get_bert(self, as_torch=False):
        if as_torch:
            return torch.from_numpy(self.bert)
        else:
            return self.bert

    def get_labels(self, as_torch=False):
        if as_torch:
            return torch.from_numpy(self.labels)
        else:
            return self.labels

    def get_dataloader(self, batch_size=32, shuffle=True):
        # Create a TensorDataset
        dataset = TensorDataset(self.get_bert(as_torch=True), self.get_labels(as_torch=True))

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
