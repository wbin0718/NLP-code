def tokenizing(df,tokenizer):
    data = tokenizer(df["title"], padding = "max_length", truncation = True, return_tensors="pt", max_length = 512)
    return data