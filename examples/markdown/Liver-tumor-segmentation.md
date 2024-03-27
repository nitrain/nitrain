# Liver tumor segmentation with nitrain

![image.png](Liver-tumor-segmentation_files/image.png)

This example shows you how to train a model to perform liver tumor segmentation using nitrain. It is a classic example of medical image segmentation. 

We will create a model with keras and do everything else (data sampling + augmentation, training, explaining results) with nitrain.

## About the data

The dataset can be downloaded from the [Liver Tumor Segmentation](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation/data) dataset on Kaggle. It is about 5 GB in size and contains 130 CT scans of the liver along with associated segmentation images where tumors have been identified.

To run this example, download the dataset ("archive.zip") and unpack it onto your desktop. Then we are ready to go!

## Creating a dataset instance

Since we have the images in a local folder and they are not in any special format, we can use the `FolderDataset` class to load them into memory. We want to use the raw images as inputs and the segmentated images as outputs.

The data looks like this:

```md
kaggle-liver-ct/
    volumes/
        volume-0.nii
        volume-1.nii
        ...
    sgementations/
        segmentation-0.nii
        segmentation-1.nii
```

Notice in particular how the images follow a specific pattern with the participant id located at the end of the file. We will use this to make sure the volumes and segmentations are correctly matched.


```python
from nitrain.datasets import FolderDataset

dataset = FolderDataset('~/Desktop/kaggle-liver-ct',
                        x={'pattern': 'volumes/volume-{id}.nii'},
                        y={'pattern': 'segmentations/segmentation-{id}.nii'})
print(dataset.x)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[1], line 3
          1 from nitrain.datasets import FolderDataset
    ----> 3 dataset = FolderDataset('~/Desktop/kaggle-liver-ct',
          4                         x={'pattern': 'volumes/volume-{id}.nii'},
          5                         y={'pattern': 'segmentations/segmentation-{id}.nii'})
          6 print(dataset.x)


    File ~/Desktop/nitrain/nitrain/datasets/folder_dataset.py:68, in FolderDataset.__init__(self, base_dir, x, y, x_transforms, y_transforms, datalad)
         65 x = [os.path.join(base_dir, file) for file in x]
         67 # GET Y
    ---> 68 participants_file = os.path.join(base_dir, y_config['file'])
         69 if participants_file.endswith('.tsv'):
         70     participants = pd.read_csv(participants_file, sep='\t')


    KeyError: 'file'





```python

```
