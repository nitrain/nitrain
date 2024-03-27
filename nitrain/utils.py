
import os
import ants
import numpy as np

def get_nitrain_dir():
    if os.environ.get('NITRAIN_DIR') is not None:
        nitrain_dir = os.environ['NITRAIN_DIR']
    else:
        nitrain_dir = os.path.join(os.path.expanduser('~'), '.nitrain')
    
    if not os.path.exists(nitrain_dir):
        os.mkdir(nitrain_dir)
        
    return nitrain_dir    

def files_to_array(files, dtype='float32'):
    # read in the images to a numpy array
    img_arrays = []
    for file in files:
        img = ants.image_read(file)
        img_array = img.numpy()
        img_arrays.append(img_array)
        
    img_arrays = np.array(img_arrays, dtype=dtype)
    return img_arrays
