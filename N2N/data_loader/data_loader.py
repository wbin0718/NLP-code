import numpy as np
from torch.utils.data import Dataset
from utils.get_char_label_sequence import get_char_label_sequence


class DPDataset(Dataset):

    def __init__(self, tokenizer, raw_data,label_vocab, max_len = 512, truncate = True, is_inference=False):
        if is_inference:
            self.data = get_char_sequence_for_eval(tokenizer,raw_data, max_len=max_len)
        else :
            self.data = get_char_label_sequence(tokenizer, raw_data, label_vocab, max_len=max_len)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        input, label = self.data[idx]
        chars, token_type_ids, attention_mask = input
        input_ids = np.array(chars)
        token_type_ids = np.array(token_type_ids)
        attention_mask = np.array(attention_mask)
        
        label = np.array(label)

        item = [input_ids ,token_type_ids, attention_mask, label]
        item = {"input_ids":input_ids, "token_type_ids":token_type_ids, "attention_mask" : attention_mask, "labels":label}
        return item