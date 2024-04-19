
from ..readers.utils import infer_reader

class Dataset:
    
    def __init__(self, inputs, outputs, transforms=None, base_dir=None):
        """
        Create a nitrain dataset.
        
        Examples
        --------
        >>> dataset = datasets.Dataset(
        ...     inputs = readers.PatternReader('~/desktop/ds004711/sub-*/anat/*_T1w.nii.gz'),
        ...     outputs = readers.ColumnReader('~/desktop/ds004711/participants.tsv', 'age'),
        ...     
        ... )
        >>> dataset = datasets.Dataset(
        ...     inputs = [img1, img2, img3, img4],
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.Dataset(
        ...     inputs = np.random.randn(10, 128, 128),
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.Dataset(
        ...     inputs = [
        ...         readers.PatternReader(),
        ...         readers.PatternReader(),
        ...     ]
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.Dataset(
        ...     inputs = {
        ...         't1': readers.PatternReader(),
        ...         't2': readers.PatternReader(),
        ...     }
        ...     outputs = readers.ColumnReader()
        ... )
        >>> dataset = datasets.Dataset(
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
        >>> dataset = datasets.Dataset(
        ...     inputs = {'anat': readers.PatternReader()},
        ...     outputs = {'seg': readers.ColumnReader()},
        ...     transforms = {
        ...         'anat': tx.RangeNormalize(0,1),
        ...         ['anat','seg']: tx.Resample((128,128,128))
        ...     }
        ... )
        """
        inputs = infer_reader(inputs)
        outputs = infer_reader(outputs)
        
        inputs.map_values(base_dir=base_dir, base_label='inputs')
        outputs.map_values(base_dir=base_dir, base_label='outputs')
                    
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
            #{'inputs': ntimage} -> img
            #{'inputs': [{'anat': ntimage}, {'func': ntimage}]} -> [img, img]
            #{'inputs': {'compose': {'anat': ntimage}, {'func': ntimage}}, {'anat-2', ntimage}} -> [[img, img], img]
            #{'inputs': {'anat': ntimage}, {'func': ntimage}, {'other': {'other-1': ntimage, 'other-2': ntimage} }} -> [img, img, [img, img]]
            x_raw = self.inputs[i]
            y_raw = self.outputs[i]
            
            if self.transforms:
                for tx_name, tx_value in self.transforms.items():
                    x_raw = apply_transforms(tx_name, tx_value, x_raw)
                    y_raw = apply_transforms(tx_name, tx_value, y_raw)
                            
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

def apply_transforms(tx_name, tx_value, x):
    #{'inputs': ntimage} -> img
    #{'inputs': {'anat': ntimage}, {'func': ntimage}} -> [img, img]
    #{'inputs': {'compose': {'anat': ntimage}, {'func': ntimage}}, {'anat-2', ntimage}} -> [[img, img], img]
    #{'inputs': {'anat': ntimage}, {'func': ntimage}, {'other': {'other-1': ntimage, 'other-2': ntimage} }} -> [img, img, [img, img]]
    
    # tx_name can be string or tuple
    if not isinstance(tx_name, tuple):
        tx_name = (tx_name,)
    
    for x_name, x_value in x.items():
        if isinstance(x_value, dict):
            if x_name in tx_name:
                pass
            else:
                apply_transforms(tx_name, tx_value, x_value)
        else:
            if x_name in tx_name:
                pass