

class BaseDataset:

    def filter(self, expr):
        raise NotImplementedError('Not implemented')

    def precompute(self):
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
            x_raw = self.x_config[i]
            y_raw = self.y_config[i]
            
            if self.x_transforms:
                for x_tx in self.x_transforms:
                    x_raw = x_tx(x_raw)
                        
            if self.y_transforms:
                for y_tx in self.y_transforms:
                    y_raw = y_tx(y_raw)
            
            x_items.append(x_raw)
            y_items.append(y_raw)
        
        if not is_slice:
            x_items = x_items[0]
            y_items = y_items[0]

        return x_items, y_items
    
    def __len__(self):
        return len(self.x)