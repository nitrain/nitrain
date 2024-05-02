
class Predictor:
    
    def __init__(self, model, task, sampler=None):
        self.model = model
        self.task = task
        self.sampler = sampler
    
    def predict(self, object, as_image=True):
        """
        Perform inference on an ntimage, nt.Dataset, or nt.Loader.
        
        The function performs inference on the supplied object using
        the fitted model intialized with the predictor. The result of
        the prediction will be one (or a sequence of) of the following
        depending on the model: ntimage, np.ndarray, scalar.
        """
        pass