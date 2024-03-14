import ants
import numpy as np
from nitrain import datasets, loaders, samplers as sp, transforms as tx

dataset = datasets.FolderDataset(base_dir='/Users/ni5875cu/Desktop/ds004711',
                                 x={'pattern': 'sub-*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                 y={'file': 'participants.tsv', 'column': 'age'},
                                 x_transforms=[tx.Resample((4,4,4), use_spacing=True)])
dataset.x = dataset.x[:7]
dataset.y = dataset.y[:7]

x_raw, y_raw = dataset[:3]

### SLICE SAMPLER ###

slice_sampler = sp.SliceSampler(sub_batch_size=24, axis=0, shuffle=True)
slice_batch = slice_sampler(x_raw, y_raw)

# the 3 3D images have now been split into lists of sliced (2D) images with length 24
# total slices: 3 images * 64 slices = 192
# total batches: 192 slices / 24 slices per batch = 8
for a, b in slice_batch:
    print(len(a))

### BLOCK SAMPLER ###

# the 3 3D images have now been split into length-24 lists of blocked (smaller 3D) images with size 32x32x24
# the total number of blocks is determined by block size and stride
block_sampler = sp.BlockSampler(block_size=(32,32,24), stride=(6,6,4), sub_batch_size=24)
block_batch = block_sampler(x_raw, y_raw)
a, b = next(iter(block_batch))

# plot a random block with the entire image
x_raw[0].plot(overlay=ants.decrop_image(a[15], x_raw[0]*0), axis=2)


#### PATCH SAMPLER ###

x = [ants.image_read(ants.get_data('r16')) for _ in range(10)]
y = list(range(10))
dataset = datasets.MemoryDataset(x, y)
x_raw, y_raw = dataset[:3]

# the 3 2D images have now been split into length-24 lists of patched (smaller 2D) images with size 64x64
# the total number of blocks is determined by patch size and stride
patch_sampler = sp.PatchSampler(patch_size=(64,64), stride=(8,8), sub_batch_size=24)
patch_batch = patch_sampler(x_raw, y_raw)

patch_iter = next(iter(patch_batch))
x_raw[0].plot(overlay=ants.decrop_image(a[10], x_raw[0]*0), axis=2)