from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc}