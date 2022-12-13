from tqdm import tqdm

def get_char_label_sequence(tokenizer, sents, label_vocab, max_len=512, truncate = True):

    cls_token = tokenizer._cls_token
    cls_token_id = tokenizer.cls_token_id
    sep_token = tokenizer._sep_token
    sep_token_id = tokenizer.sep_token_id
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    unk_token = tokenizer._unk_token
    unk_token_id = tokenizer.unk_token_id

    data = []
    for idx, a_sent in tqdm(sents.iterrows()):
        chars = []
        labels = []

        for char, label in zip(a_sent["sentence"], a_sent["labels"]):
            char_idx = tokenizer.convert_tokens_to_ids(char)
            label_idx = label_vocab[label]

            chars.append(char_idx)
            labels.append(label_idx)

        assert len(chars) == len(labels), "INVALID DATA LENGTH: N2N must get SAME INPUT and OUTPUT LENGTH"

        if truncate == True :
            chars = chars[:max_len - 2]
            lagels = labels[:max_len - 2]
        
        chars = [cls_token_id] + chars + [sep_token_id]
        labels = [label_vocab["[PAD]"]] + labels + [label_vocab["[PAD]"]]

        attention_mask = [1] * len(chars)
        N = max_len - len(chars)
        chars = chars + [pad_token_id] * N
        token_type_ids = [0] * len(chars)
        attention_mask = attention_mask + [0] * N
        labels = labels + [label_vocab["[PAD]"]] * N

        data.append(((chars, token_type_ids, attention_mask), labels))
    return data