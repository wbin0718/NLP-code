import torch
import torch.nn as nn
def sentiment_predict(model, tokenizer,new_sentence) :

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    tokens = bos_token + new_sentence + eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item = tokenizer(tokens, padding="max_length", max_length = 128, truncation  =True, return_tensors="pt")
    item = item["input_ids"]
    item = item.to(device)
    logits = model(item)
    softmax = nn.Softmax()
    logits = softmax(logits)
    logits = torch.max(logits, dim = 1)

    if logits[1].item() == 1 :
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(logits[0].item() * 100))
    else :
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format(logits[0].item() * 100))

