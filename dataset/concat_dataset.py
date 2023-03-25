import bisect
import warnings
import torch

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.distribution_str = []
        self.num_distribution = len(datasets)
        self.distribution_counts = []
        for i in range(self.num_distribution):
            self.distribution_counts.append(len(datasets[i]))
            self.distribution_str.append(datasets[i].dataset_name)

        self.distribution_counts = torch.FloatTensor(self.distribution_counts)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        # dataset idx: 0 for clean, 1 for robust
        return self.datasets[dataset_idx][sample_idx], dataset_idx # return this if using the dataaset idx.

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

