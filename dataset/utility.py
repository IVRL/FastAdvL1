import numpy as np

import torch
from torch.utils.data import SubsetRandomSampler, Sampler

class SubsetSampler(Sampler):

    def __init__(self, indices):

        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class HybridBatchSampler(Sampler):

    def __init__(self, idx4ori, idx4plus, plus_prop, batch_size, permutation):

        self.idx4ori = idx4ori
        self.idx4plus = idx4plus
        self.plus_prop = plus_prop
        self.batch_size = batch_size
        self.permutation = permutation

        self.use_plus = plus_prop > 0

    def __iter__(self,):

        batch = []
        # No additional data
        if self.use_plus is False:
            if self.permutation is True:
                self.idx4ori = np.random.permutation(self.idx4ori)
            for idx in self.idx4ori:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
                batch = []
        else:
            max_num_plus = int(self.batch_size * self.plus_prop)
            max_num_ori = int(self.batch_size - max_num_plus)
            idx_in_ori = 0
            if self.permutation is True:
                self.idx4plus = np.random.permutation(self.idx4plus)
            for idx in self.idx4plus:
                batch.append(idx)
                if len(batch) == max_num_plus:
                    if len(self.idx4ori[idx_in_ori: idx_in_ori + max_num_ori]) != max_num_ori:
                        if self.permutation is True:
                            self.idx4ori = np.random.permutation(self.idx4ori)
                        idx_in_ori = 0
                    batch = batch + [v for v in self.idx4ori[idx_in_ori: idx_in_ori + max_num_ori]]
                    idx_in_ori += max_num_ori
                    yield batch
                    batch = []
            if len(batch) > 0:
                num_ori = int(len(batch) / self.plus_prop * (1. - self.plus_prop))
                if len(self.idx4ori[idx_in_ori: idx_in_ori + num_ori]) != num_ori:
                    if self.permutation is True:
                        self.idx4ori = np.random.permutation(self.idx4ori)
                    idx_in_ori = 0
                batch = batch + [v for v in self.idx4ori[idx_in_ori: idx_in_ori + num_ori]]
                idx_in_ori += num_ori
                yield batch
                batch = []

    def __len__(self,):

        if self.use_plus is False:
            return len(self.idx4ori - 1) // self.batch_size + 1
        else:
            max_num_plus = int(self.batch_size * self.plus_prop)
            return len(self.idx4plus - 1) // max_num_plus + 1



