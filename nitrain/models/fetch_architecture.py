

def fetch_architecture(name):
    """
    Fetch an architecture function from ANTsPyNet. Architectures
    are callable functions that allow you to flexibly create 
    deep learning models according to your needs.
    
    Returns
    -------
    An instance of the ModelArchitecture class with many 
    convenience functions. An actual model can be created
    via the .create() method.
    """
    pass


class ModelArchitecture:
    
    def __init__(self, fn):
        pass
    
    def create(self, 
               input_size=None, 
               output_size=None, 
               framework='default', 
               **kwargs):
        pass