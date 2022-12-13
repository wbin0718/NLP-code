import torch.nn as nn
from transformers import Trainer

class Trainer(Trainer):
    def __init__(self, label_vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_vocab = label_vocab
        self.ignore_label_idx = self.label_vocab["[PAD]"]
    
    def compute_loss(self,model, inputs, return_outputs = False):

        labels = inputs.pop("labels")
        
        step_logits = model(input_ids = inputs["input_ids"].long(), token_type_ids = inputs["token_type_ids"].long(), attention_mask = inputs["attention_mask"].long())

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.ignore_label_idx)
        
        B, S, C = step_logits.shape

        predicted = step_logits.view(-1,C)
        reference = labels.view(-1)

        loss = self.criterion(predicted, reference.long())

        return (loss, step_logits) if return_outputs else loss