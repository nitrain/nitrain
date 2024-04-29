

from ..readers.utils import infer_reader
from .utils import reduce_to_list, apply_transforms

__all__ = ['GCSDataset']

class GCSDataset:
    
    def __init__(self, bucket, inputs, outputs, transforms=None, base_dir=None, base_file=None, credentials=None):
        """
        Create a nitrain dataset.
        
        Examples
        --------
        import nitrain as nt
        from nitrain import readers
        d = nt.GCSDataset(
            inputs=readers.PatternReader('sub-*/anat/*_T1w.nii.gz'),
            outputs=readers.ColumnReader('participants.tsv', 'age'),
            base_dir='datasets/nick-2/ds004711',
            bucket='ants-dev'
        )
        d2 = nt.Dataset(
            inputs=readers.PatternReader('sub-*/anat/*_T1w.nii.gz'),
            outputs=readers.ColumnReader('age','participants.tsv'),
            base_dir='~/Desktop/ds004711/'
        )
        """
        inputs = infer_reader(inputs)
        #outputs = infer_reader(outputs)
        
        inputs.map_values(base_dir=base_dir, base_file=base_file, base_label='inputs', bucket=bucket, credentials=credentials)
       # outputs.map_values(bucket, credentials, base_dir=base_dir, base_file=base_file, base_label='outputs')
                    
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
                    x_raw, y_raw = apply_transforms(tx_name, tx_value, x_raw, y_raw)
                    
            x_raw = reduce_to_list(x_raw)
            y_raw = reduce_to_list(y_raw)
            
            x_items.append(x_raw)
            y_items.append(y_raw)
        
        if not is_slice:
            x_items = x_items[0]
            y_items = y_items[0]

        return x_items, y_items
    
    def __len__(self):
        return len(self.inputs)
    
    def __str__(self):
        raise NotImplementedError('Not implemented')
    
    def __repr__(self):
        s = 'GCSDataset (n={})\n'.format(len(self))
        
        s = s +\
            '     {:<10} : {}\n'.format('Inputs', self.inputs)+\
            '     {:<10} : {}\n'.format('Outputs', self.outputs)+\
            '     {:<10} : {}\n'.format('Transforms', len(self.transforms) if self.transforms else '{}')
        return s