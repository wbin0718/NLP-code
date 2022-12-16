from tqdm import tqdm
def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, data_labels = [],[]

    for example, label in tqdm(zip(examples,labels), total = len(examples)):

        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        tokens = bos_token + example + eos_token

        tokens = tokenizer(tokens, padding="max_length", max_length = max_seq_len, truncation = True)
        input_id = tokens["input_ids"]

        input_ids.append(input_id)
        data_labels.append(label)
    
    return input_ids, data_labels
        