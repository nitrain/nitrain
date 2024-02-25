# Full nitrain example of training a convolutional neural network (AlexNet)
# on T1 MRI images with a binary outcome (gender) using Keras
#
# This example gives a good overview of main topics in nitrain:
# - data loading + augmentation
# - high-level training
# - visualization + explainability
import nitrain

data = nitrain.get_data('ds004711', return_data=True)