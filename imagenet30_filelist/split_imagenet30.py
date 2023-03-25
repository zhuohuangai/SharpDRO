# split imagenet30 dataset in to:
# 5 subsets which follows the specified poisson distribution.
import numpy as np
from scipy.stats import poisson
import math

paths = []
targets = []
with open('./imagenet30_filelist/imagenet30.txt') as f:
    for line in f.readlines():
        p, l = line.split()
        paths.append(p)
        targets.append(int(l))

targets = np.array(targets)
num_classes = (targets).max()

severity_level = np.arange(0, 6)
probs = poisson.pmf(severity_level, 1)
num_per_distribution = []
idx_list = []
for prob in probs:
    num_per_distribution.append(math.floor(1000 * prob))
    idx_list.append([])
num_slices = [0]
num_slices = num_slices + list(np.cumsum(num_per_distribution))
val_idx = []
for i in range(30):
    idx = np.where(targets == i)[0]
    idx = np.random.choice(idx, 1000, False)
    for j in range(6):
        idx_list[j].extend(idx[num_slices[j]:num_slices[j+1]])

val_idx = np.array(range(len(targets)))
for i in range(6):
    val_idx = [idx for idx in val_idx if idx not in idx_list[i]]

for i in range(6):
    for j in idx_list[i]:
        with open(f'imagenet30_distribution_{i:d}.txt', 'a') as f:
            f.writelines(paths[j] + ' ' + str(targets[j]) + '\n')

for i in val_idx:
    with open('imagenet30_val.txt', 'a') as f:
        f.writelines(paths[i] + ' ' + str(targets[i]) + '\n')
