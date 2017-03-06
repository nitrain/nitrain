"""
This example shows how to use stratified sampling to have an equal number
of samples from each class in each batch
"""
import torch
import numpy as np
from torchsample import StratifiedSampler
y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
sampler = StratifiedSampler(class_vector=y, batch_size=2)
for i, batch_idx in enumerate(sampler):
    print(batch_idx, ' - ' , y[batch_idx])
    if (i+1) % 2 == 0:
        print('\n')
#[out]:
#7  -  1
#1  -  0

#0  -  0
#3  -  1

#4  -  0
#2  -  1

#5  -  0
#6  -  1

#You can see that it evenly samples each batch from 
#the 0 and 1 classes. To use it in a sampler:
from torchsample import TensorDataset
x = torch.randn(8,2)
y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1]))

loader = TensorDataset(x, y, batch_size=2, sampler='stratified')

# OR:

x = torch.randn(8,2)
y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
sampler = StratifiedSampler(y, batch_size=2)
loader = TensorDataset(x, y, batch_size=2, sampler=sampler)