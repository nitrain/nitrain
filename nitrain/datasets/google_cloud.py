


from .utils import reduce_to_list, apply_transforms
from .dataset import Dataset
from ..readers.utils import infer_reader

__all__ = ['GoogleCloudDataset']

class GoogleCloudDataset(Dataset):
    
    def __init__(self, bucket, inputs, outputs, transforms=None, base_dir=None, base_file=None, credentials=None):
        """
        Create a nitrain dataset.
        
        Examples
        --------
        import nitrain as nt
        from nitrain import readers
        d = nt.GoogleCloudDataset(
            inputs=readers.PatternReader('sub-*/anat/*_T1w.nii.gz'),
            outputs=readers.ColumnReader('age', 'participants.tsv'),
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
        outputs = infer_reader(outputs)
        
        inputs.map_gcs_values(base_dir=base_dir, base_file=base_file, base_label='inputs', bucket=bucket, credentials=credentials)
        outputs.map_gcs_values(bucket=bucket, credentials=credentials, base_dir=base_dir, base_file=base_file, base_label='outputs')
                    
        self.inputs = inputs
        self.outputs = outputs
        self.transforms = transforms
    
    def __repr__(self):
        s = 'GoogleCloudDataset (n={})\n'.format(len(self))
        
        s = s +\
            '     {:<10} : {}\n'.format('Inputs', self.inputs)+\
            '     {:<10} : {}\n'.format('Outputs', self.outputs)+\
            '     {:<10} : {}\n'.format('Transforms', len(self.transforms) if self.transforms else '{}')
        return s