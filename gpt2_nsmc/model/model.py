import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self,model_name,tokenizer):

        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)

    
    def forward(self,input_ids):

        outputs = self.model(input_ids = input_ids)
        cls_token = outputs[0][:,-1]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)

        return logits