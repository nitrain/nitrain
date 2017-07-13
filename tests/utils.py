
import numpy as np
import torch as th

def get_test_data(num_train=1000, num_test=500, 
                  input_shape=(10,), output_shape=(2,),
                  classification=True, num_classes=2):
    """Generates test data to train a model on.

    classification=True overrides output_shape
    (i.e. output_shape is set to (1,)) and the output
    consists in integers in [0, num_class-1].

    Otherwise: float output with shape output_shape.
    """
    samples = num_train + num_test
    if classification:
        y = np.random.randint(0, num_classes, size=(samples,))
        X = np.zeros((samples,) + input_shape)
        for i in range(samples):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((samples,))
        X = np.zeros((samples,) + input_shape)
        y = np.zeros((samples,) + output_shape)
        for i in range(samples):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (th.from_numpy(X[:num_train]), th.from_numpy(y[:num_train])), \
           (th.from_numpy(X[num_train:]), th.from_numpy(y[num_train:]))