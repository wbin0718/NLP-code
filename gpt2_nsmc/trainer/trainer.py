import torch.nn as nn
from transformers import Trainer

class Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self,model, inputs, return_outputs = False):

        labels = inputs.pop("labels")
        
        step_logits = model(input_ids = inputs["input_ids"].long())

        self.criterion = nn.CrossEntropyLoss()
        
        B, C = step_logits.shape

        predicted = step_logits.view(-1,C)
        reference = labels.view(-1)

        loss = self.criterion(predicted, reference.long())

        return (loss, step_logits) if return_outputs else loss