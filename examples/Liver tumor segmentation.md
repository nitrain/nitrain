# Liver tumor segmentation with nitrain

![image.png](Liver tumor segmentation_files/image.png)

This example shows you how to train a model to perform liver tumor segmentation using nitrain. It is a classic example of medical image segmentation.

We will create a model with keras and do everything else (data sampling + augmentation, training, explaining results) with nitrain.

![png](Liver-tumor-segmentation_files/image.png)

## About the data

The dataset can be downloaded from the [Liver Tumor Segmentation](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation/data) dataset on Kaggle. It is about 5 GB in size and contains 130 CT scans of the liver along with associated segmentation images where tumors have been identified.

To run this example, download the dataset ("archive.zip") and unpack it onto your desktop. Then we are ready to go!

## Creating a dataset instance

Since we have the images in a local folder and they are not in any special format, we can use the `FolderDataset` class to load them into memory. We want

```python

```
