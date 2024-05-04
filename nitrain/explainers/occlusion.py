

class OcclusionExplainer:
    """
    Create an occlusion explainer which determines the spatial importance
    of a medical image for model predictions by systematically setting parts
    of the image to zero and determining how the model prediction changes.
    
    This explainer can be run with or without ground-truth labels. Without ground
    truth labels, importance will be determined by how much the predicted result changes
    with the occluded image compared to the original image. With ground truth labels, 
    importance will be determined by how much the performance of the model changes (although
    the change in predicted result will also be available in this case).
    """
    
    def __init__(self, model, sampler=None):
        """
        Initialize an occlusion explainer from a fitted model.
        """
        self.model = model
        self.sampler = sampler
        
        # generated once fit() method is called
        self.result_image = None
        
    def fit(self, dataset):
        """
        Run occlusion explainer on a dataset loader, a single image, or
        a list of images.
        """
        return 1