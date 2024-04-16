

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
                        # TODO: match to name in config
                        pass
                            
            x_items.append(x_raw)
            y_items.append(y_raw)
        
        if not is_slice:
            x_items = x_items[0]
            y_items = y_items[0]

        return x_items, y_items
    
    def __len__(self):
        return len(self.x)