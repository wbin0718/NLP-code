from torch.utils.data import Dataset

class Dataset(Dataset) :
    
    def __init__(self, inputs, labels):

        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):

        return {"input_ids" : self.inputs[idx], "labels":self.labels[idx]}