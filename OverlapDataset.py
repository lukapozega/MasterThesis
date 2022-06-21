import torch
from torch.utils.data import Dataset
from data import encode_cigar
from sklearn.utils import class_weight
import numpy as np

class OverlapDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
        self.class_weights = torch.tensor(class_weights,dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [x[0] for x in self.X[idx][0]], [x[0] for x in self.X[idx][0]], self.X[idx][1], self.X[idx][2], self.y[idx]

    def from_file(dataset, extended_cigar, fn):
        with open(dataset, "r") as f:
            lines = f.readlines()
            X = []
            y = []
            p, n = 0, 0
            for line in lines:
                s = line.strip().split()
                try:
                    l = int(s[3])
                except IndexError as e:
                    print(s)
                    exit()
                if (not fn and l == 2):
                    continue
                X.append((encode_cigar(s[2], extended_cigar), int(s[0]), float(s[1])))
                y.append(l)
        return X, y
