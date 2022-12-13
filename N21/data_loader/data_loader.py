from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        data = {k:v[idx] for k,v in self.input_data.items()}
        data["labels"] = self.labels[idx]
        return data