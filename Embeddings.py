"""
This file contains the Embeddings class, which is used to load the embeddings from the .npy files

By: David Pogrebitskiy and Jacob Ostapenko
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle


class Embeddings:
    def __init__(self, doc2vec_file, bert_file, raw_labels_file, encoded_labels_file, encoder_pickle):
        # Load the embeddings from the .npy files
        self.doc2vec = np.load(doc2vec_file, allow_pickle=True)
        self.bert = np.load(bert_file, allow_pickle=True)
        self.raw_labels = np.load(raw_labels_file, allow_pickle=True)
        self.encoded_labels = np.load(encoded_labels_file, allow_pickle=True)
        with open(encoder_pickle, 'rb') as f:
            self.encoder = pickle.load(f)

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

    def get_labels(self, as_torch=False, raw=False):
        labels = self.raw_labels if raw else self.encoded_labels
        if as_torch:
            return torch.from_numpy(labels)
        else:
            return labels

    def decode_labels(self, encoded_labels):
        return self.encoder.inverse_transform(encoded_labels)
