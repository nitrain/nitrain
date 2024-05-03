
class Predictor:
    
    def __init__(self, model, task, sampler=None):
        self.model = model
        self.task = task
        self.sampler = sampler
    
    def predict(self, dataset):
        """
        Perform inference on an nt.Dataset.
        
        The function performs inference on the supplied object using
        the fitted model intialized with the predictor. The object
        will go through the sampler before inference but will maintain
        its original shape.
        
        The task determines whether the resulting inference is converted
        to an image or not and whether the prediction values are rounded
        to be in the same style as the dataset output.
        
        The result of the prediction will be one (or a sequence of) of the 
        following depending on the model: ntimage, np.ndarray, scalar.
        """
        pass