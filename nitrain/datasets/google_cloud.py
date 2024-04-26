

class GoogleCloudDataset:
    
    def __init__(self, bucket, inputs, outputs, transforms=None, base_dir=None, credentials=None, fuse=False):
        """
        Create a nitrain dataset from images stored on google cloud.
        
        Examples
        --------
        >>> dataset = datasets.GoogleCloudDataset(
        ...     inputs = readers.PatternReader(),
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.GoogleCloudDataset(
        ...     inputs = [
        ...         readers.PatternReader(),
        ...         readers.PatternReader(),
        ...     ]
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.GoogleCloudDataset(
        ...     inputs = {
        ...         't1': readers.PatternReader(),
        ...         't2': readers.PatternReader(),
        ...     }
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.GoogleCloudDataset(
        ...     inputs = readers.PatternReader(),
        ...     outputs = readers.ColumnReader(),
        ...     transforms = {
        ...         'inputs': tx.RangeNormalize(0,1),
        ...         ['inputs','outputs']: [
        ...             tx.Resample((128,128,128)),
        ...             tx.Reorient('RPI')
        ...         ]
        ...     }
        ... )
        >>> dataset = datasets.GoogleCloudDataset(
        ...     inputs = {'anat': readers.PatternReader()},
        ...     outputs = {'seg': readers.ColumnReader()},
        ...     transforms = {
        ...         'anat': tx.RangeNormalize(0,1),
        ...         ['anat','seg']: tx.Resample((128,128,128))
        ...     }
        ... )
        """
        self.bucket = bucket
        self.base_dir = base_dir
        self.credentials = credentials
        self.fuse = fuse
        
        self.inputs = inputs
        self.outputs = outputs
        self.transforms = transforms

    def filter(self, expr):
        raise NotImplementedError('Not implemented')
    
    def prefetch(self):
        raise NotImplementedError('Not implemented')
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = list(range(idx.stop)[idx])
            is_slice = True
        else:
            idx = [idx]
            is_slice = False
            
        x_items = []
        y_items = []
        for i in idx:
            x_raw = self.inputs(i)
            y_raw = self.outputs(i)
            
            if self.transforms:
                for tx_name, tx_list in self.transforms:
                    if tx_name == 'x':
                        for tx_fn in tx_list:
                            x_raw = tx_fn(x_raw)
                    elif tx_name == 'y':
                        for tx_fn in tx_list:
                            y_raw = tx_fn(y_raw)
                    elif tx_name == 'co':
                        for tx_fn in tx_list:
                            x_raw, y_raw = tx_fn(x_raw, y_raw)
                    else:
                        # TODO: match to name in inputs / outputs
                        pass
                            
            x_items.append(x_raw)
            y_items.append(y_raw)
        
        if not is_slice:
            x_items = x_items[0]
            y_items = y_items[0]

        return x_items, y_items
    
    def __len__(self):
        raise NotImplementedError('Not implemented')
    
    def __str__(self):
        raise NotImplementedError('Not implemented')
    
    def __repr__(self):
        raise NotImplementedError('Not implemented')