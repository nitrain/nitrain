# An example of how to download a dataset

import bids
from nitrain.utils import download_data

# will go to downloads directory
download_data('ds003826')

# get some data
# ds = dl.Dataset(localizer_path)

# todo: use bids instead of glob
# file_list = glob.glob(os.path.join(localizer_path, 'derivatives', 'fmriprep', '*', 'func', '*task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))

# result = ds.get(file_list[0])



# https://dartbrains.org/content/Download_Data.html