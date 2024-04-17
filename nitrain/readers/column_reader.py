import os
import pandas as pd
import ntimage as nt


class ColumnReader:
    def __init__(self, file, column, is_image=False):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from tempfile import NamedTemporaryFile
        >>> import pandas as pd
        >>> from nitrain.readers import ColumnReader
        >>> df = pd.DataFrame.from_dict({'x':[1,2,3], 'y':['a','b','c']})
        >>> file = NamedTemporaryFile(suffix='.csv')
        >>> df.to_csv(file.name)
        >>> reader = ColumnReader(file.name, 'x')
        >>> value = reader[1] # 'b'
        """
        self.file = file
        self.column = column
        self.is_image = is_image
    
    def map_values(self, base_dir=None):
        file = self.file
        column = self.column
        is_image = self.is_image

        if base_dir is not None:
            file = os.path.join(base_dir, file)
        
        if not os.path.exists(file):
            raise Exception(f'No file found at {file}')
        
        if file.endswith('.tsv'):
            participants = pd.read_csv(file, sep='\t')
        elif file.endswith('.csv'):
            participants = pd.read_csv(file)
            
        values = participants[column].to_numpy()
        
        if id is not None:
            ids = list(participants[id].to_numpy())
        else:
            ids = None
        
        self.values = values
        self.ids = ids
        self.file = file
        self.column = column
        self.is_image = is_image

    def __getitem__(self, idx):
        value = self.values[idx]
        if self.is_image:
            value = nt.load(value)
        return value
