

from ptsample.transforms import ToTensor, Compose, RangeNormalize, Slice2D
from ptsample import DataLoader
from ptsample import FolderDataset

def example_RangeNormalize():
	tform = Compose([ToTensor(), RangeNormalize(-1,1)])
	data = FolderDataset(root='~/desktop/data/segmentation/skullstrip_hcp_2mm_npy/', 
		class_mode='image', input_regex='*img*', target_regex='*mask*', 
		loader='npy', transform=tform)
	loader = iter(DataLoader(data))
	x,y = loader.next()
	print(x.min()) # -1
	print(x.max()) # +1

def example_Slice():

	tform = Compose([ToTensor(), Slice2D(axis=0)])
	data = FolderDataset(root='/users/ncullen/desktop/data/segmentation/skullstrip_hcp_2mm_npy/', 
		class_mode='image', input_regex='*img*', target_regex='*mask*', 
		loader='npy', co_transform=tform)
	loader = iter(DataLoader(data))
	x,y = loader.next()
	print(x.size()) # (109, 91) -> now 2d instead of 3d
	print(y.size()) # (109, 91) -> now 2d instead of 3d