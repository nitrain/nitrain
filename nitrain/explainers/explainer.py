

class Explainer:
    """
    Explainers help you run different experiments or algorithms on a trained
    deep learning model to better understand what is important to the model
    in the context of the medical images used for training.
    
    Explainers are also useful for generating publication-ready figures from
    trained models.
    
    Examples
    --------
    >>> model = models.fetch_pretrained('nick/t1-brain-age')
    >>> image = ants.image_read(ants.get_data('mni'))
    >>> explainer = explainers.Explainer(model, method='occlusion')
    >>> explainer.fit(image)
    """
    def __init__(self, model):
        self.model = model