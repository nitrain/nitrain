
from pathlib import Path
import ants
import numpy as np

def get_nitrain_dir():
    downloads_path = str(Path.home() / ".nitrain/")
    return downloads_path    

def files_to_array(files, dtype='float32'):
    # read in the images to a numpy array
    img_arrays = []
    for file in files:
        img = ants.image_read(file)
        img_array = img.numpy()
        img_arrays.append(img_array)
        
    img_arrays = np.array(img_arrays, dtype=dtype)
    return img_arrays