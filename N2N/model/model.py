import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, model_name, label_vocab):
        super().__init__()
        
        self.label_vocab = label_vocab
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name) 
        self.to_class = nn.Linear(self.model.config.hidden_size, len(self.label_vocab))
        
    def forward(self,input_ids, token_type_ids, attention_mask):
        outputs = self.model(input_ids, token_type_ids, attention_mask)
        outputs = outputs["last_hidden_state"]
        step_label_logits = self.to_class(outputs)
        return step_label_logits