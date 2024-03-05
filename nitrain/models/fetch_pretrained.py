

def fetch_pretrained(name):
    """
    Fetch a pretrained model from ANTsPyNet. Pretrained
    models can be used to make predictions (inference) on
    your data or as a starting point for fine-tuning to help
    improve model fitting on your data.
    
    Returns
    -------
    An instance of the PretrainedModel class that includes many
    convenience functions. The actual model can be accessed via
    the .model property.
    """
    pass


class PretrainedModel:
    
    def __init__(self, name, model):
        pass
    
    def fine_tune(self, data):
        pass