
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
            x_raw = self.inputs[i]
            y_raw = self.outputs[i]
            
            if self.transforms:
                for tx_name, tx_value in self.transforms.items():
                    # TODO: support transforms applied to input + output together
                    x_raw = apply_transforms(tx_name, tx_value, x_raw)
                    y_raw = apply_transforms(tx_name, tx_value, y_raw)
                    
            x_raw = reduce_to_list(x_raw)
            y_raw = reduce_to_list(y_raw)
            
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


def reduce_to_list(d, idx=0):
    result = []
    for key, val in d.items():
        if isinstance(val, dict):
            result.extend([reduce_to_list(d[key], idx+1)])
        else:
            result.append(val)
        
    return result if len(result) > 1 else result[0]


def apply_transforms(tx_name, tx_value, inputs, force=False):
    """
    Apply transforms recursively based on match between transform name
    and input label. If there is a match and the input is nested, then all
    children of that input will receive the transform.
    """
    if not isinstance(tx_name, tuple):
        tx_name = (tx_name,)
    if not isinstance(tx_value, list):
        tx_value = list(tx_value)
    
    for input_name, input_value in inputs.items():
        
        if isinstance(input_value, dict):
            if input_name in tx_name:
                apply_transforms(tx_name, tx_value, input_value, force=True)
            else:
                apply_transforms(tx_name, tx_value, input_value, force=force)
        else:
            if (input_name in tx_name) | force:
                for tx_fn in tx_value:
                    inputs[input_name] = tx_fn(inputs[input_name])
    
    return inputs
            