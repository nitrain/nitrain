"""
Test ModuleTrainer class

Considerations
--------------
- Inputs (#, type)
- Outputs (#, type)
- Losses
- Optimizer
- regularizers
- initializers
- constraints
- metrics

Main Functions
--------------
- compile()
- fit()
- fit_loader()
- fit_batch()
- predict()
- predict_loader()
- predict_batch()
- evaluate()
- evaluate_loader()
- evaluate_batch()

Utility Functions
-----------------
- summary()
- save()
"""
# standard imports
import os
from torchvision import datasets

# torch imports
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# torchsample imports
from torchsample.modules import ModuleTrainer


# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

### SETUP DATA + MODULES ###
def load_mnist(root='~/desktop/data/mnist', nb_train=10000, nb_test=1000):
    """
    loads and processes mnist data
    """
    if root.startswith('~'):
        root = os.path.expanduser(root)

    dataset = datasets.MNIST(root, train=True, download=True)
    x_train, y_train = th.load(os.path.join(dataset.root, 'processed/training.pt'))
    x_test, y_test = th.load(os.path.join(dataset.root, 'processed/test.pt'))

    x_train = x_train.float()
    y_train = y_train.long()
    x_test = x_test.float()
    y_test = y_test.long()

    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)

    # only train on a subset
    x_train = x_train[:nb_train]
    y_train = y_train[:nb_train]
    x_test = x_test[:nb_test]
    y_test = y_test[:nb_test]

    return (x_train, y_train), (x_test, y_test)

def load_models():
    class MNIST_MODEL_in1out1(nn.Module):
        def __init__(self):
            super(MNIST_MODEL_in1out1, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1600, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1600)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    class MNIST_MODEL_in1out2(nn.Module):
        def __init__(self):
            super(MNIST_MODEL_in1out2, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1600, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1600)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x), F.log_softmax(x)

    class MNIST_MODEL_in2out1(nn.Module):
        def __init__(self):
            super(MNIST_MODEL_in2out1, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1600, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x, x2):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1600)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    class MNIST_MODEL_in2out2(nn.Module):
        def __init__(self):
            super(MNIST_MODEL_in2out2, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1600, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x, x2):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 1600)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)

            x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
            x2 = F.relu(F.max_pool2d(self.conv2(x2), 2))
            x2 = x2.view(-1, 1600)
            x2 = F.relu(self.fc1(x2))
            x2 = F.dropout(x2, training=self.training)
            x2 = self.fc2(x2)
            return F.log_softmax(x), F.log_softmax(x2)

    return (MNIST_MODEL_in1out1, 
            MNIST_MODEL_in1out2,
            MNIST_MODEL_in2out1, 
            MNIST_MODEL_in2out2)

# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------


def test_module_trainer_mnist_fit(verbose=True):
    """
    Test ModuleTrainer with multiple inputs/outputs
    """
    if verbose:
        print('Running Fit tests')

    (x_train, y_train), (x_test, y_test) = load_mnist()
    m_i1o1, m_i1o2, m_i2o1, m_i2o2 = load_models()

    model_tests = {
        'm_i1o1': ([x_train,y_train], m_i1o1()),
        'm_i1o2': ([x_train,[y_train,y_train]], m_i1o2()),
        'm_i2o1': ([[x_train,x_train],y_train], m_i2o1()),
        'm_i2o2': ([[x_train,x_train],[y_train,y_train]], m_i2o2())
    }

    SUCCESSES = []
    FAILURES = []
    for key, (train_data, model) in model_tests.items():
        trainer = ModuleTrainer(model)
        trainer.compile(loss='nll_loss', optimizer='adam')
        try:
            trainer.fit(train_data[0], 
                        train_data[1], 
                        nb_epoch=1,
                        verbose=0)
            SUCCESSES.append(key)
        except:
            FAILURES.append(key)

def test_module_trainer_mnist_predict(verbose=True):
    """
    Test ModuleTrainer with multiple inputs/outputs
    """
    if verbose:
        print('Running Predict tests')

    (x_train, y_train), (x_test, y_test) = load_mnist()
    m_i1o1, m_i1o2, m_i2o1, m_i2o2 = load_models()

    model_tests = {
        'm_i1o1': ([x_train,y_train], m_i1o1),
        'm_i1o2': ([x_train,[y_train,y_train]], m_i1o2),
        'm_i2o1': ([[x_train,x_train],y_train], m_i2o1),
        'm_i2o2': ([[x_train,x_train],[y_train,y_train]], m_i2o2)
    }

    SUCCESSES = []
    FAILURES = []
    for key, (train_data, model) in model_tests.items():
        trainer = ModuleTrainer(model())
        trainer.compile(loss='nll_loss', optimizer='adam')
        try:
            trainer.predict(train_data[0], verbose=0)
            SUCCESSES.append(key)
        except:
            FAILURES.append(key)

def test_module_trainer_mnist_evaluate(verbose=True):
    """
    Test ModuleTrainer with multiple inputs/outputs
    """
    if verbose:
        print('Running Evaluate tests')

    (x_train, y_train), (x_test, y_test) = load_mnist()
    m_i1o1, m_i1o2, m_i2o1, m_i2o2 = load_models()

    model_tests = {
        'm_i1o1': ([x_train,y_train], m_i1o1),
        'm_i1o2': ([x_train,[y_train,y_train]], m_i1o2),
        'm_i2o1': ([[x_train,x_train],y_train], m_i2o1),
        'm_i2o2': ([[x_train,x_train],[y_train,y_train]], m_i2o2)
    }

    SUCCESSES = []
    FAILURES = []
    for key, (train_data, model) in model_tests.items():
        trainer = ModuleTrainer(model())
        trainer.compile(loss='nll_loss', optimizer='adam')
        try:
            trainer.evaluate(train_data[0], verbose=0)
            SUCCESSES.append(key)
        except:
            FAILURES.append(key)


if __name__=='__main__':
    test_module_trainer_mnist_fit()
    test_module_trainer_mnist_predict()
    test_module_trainer_mnist_evaluate()





