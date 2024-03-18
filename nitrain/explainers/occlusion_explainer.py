

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
    
    def __init__(self, model):
        """
        Initialize an occlusion explainer from a fitted model.
        
        Arguments
        ---------
        model : a fitted model
            The model to use when running the explainer. 
        
        Examples
        --------
        >>> model = models.fetch_pretrained('nick/t1-brain-age')
        >>> image = ants.image_read(ants.get_data('mni'))
        >>> trainer = ModelTrainer(model)
        >>> trainer.fit(loader)
        >>> explainer = explainers.OcclusionExplainer(trainer.model)
        >>> explainer.fit(image)
        >>> ants.plot(image, overlay=explainer.result_image)
        """
        self.model = model
        
        # generated once fit() method is called
        self.result_image = None
        
    def fit(self, inputs, outputs=None):
        """
        Run occlusion explainer on a dataset loader, a single image, or
        a list of images.
        
        Arguments
        ---------
        inputs : a nitrain loader, ants image, or list of ants images
            These are the images you want to run the explainer on. If you
            provide multiple images, an explained image averaged across
            all inputs will be available.
        
        outputs : numpy array, image, or list of images (optional)
            This is the ground truth labels for the inputs. If you supply
            ground truth labels then the explainer will use the change in
            model performance to determine importance rather than only change
            in magnitude of raw model prediction.
        """
        pass