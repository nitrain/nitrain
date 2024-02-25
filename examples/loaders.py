# Full example of training a convolutional neural network (AlexNet)
# on T1 MRI images with a continuous outcome (age) using Keras
#
# This example gives a good overview of main topics in nitrain:
# - data loading + augmentation
# - high-level training
# - visualization + explainability

from nitrain import utils, data
import keras

# download and load the data into memory
ds = utils.fetch_datalad('ds004711')
dataset = data.FileDataset(path = ds.path, 
                           layout = 'bids',
                           meta = 'participants.tsv',
                           x_config = {'suffix': 'T1w'},
                           y_config = {'filename': 'participants.tsv', 'column': 'age'})
dataset = dataset.filter('age < 80')
x, y = dataset.load(n=10)

# create a data loader from an in-memory dataset
loader = data.DatasetLoader(
    dataset = data.MemoryDataset(x, y)
)

# create the model
num_classes = 2
input_shape = loader.input_shape

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.fit(loader)