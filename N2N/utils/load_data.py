import pandas as pd

def load_data(fn):
    sents = []
    labels = []

    with open(fn, "r", encoding="utf-8") as f :
        sent = []
        label_sequence= []
        for line in f :
            if line[0:2] == "##":
                continue

            if line == "\n":
                sents.append({"sentence":sent[:-1],
                "labels":label_sequence[:-1]})
            
                sent = []
                label_sequence = []

            else :
                index, word, lemma, pos, head, dp_tag = line.rstrip().split("\t")
                for i,c in enumerate(word):
                    sent.append(c)
                    if i ==0 :
                        label_sequence.append(f"B-{dp_tag}")
                    else :
                        label_sequence.append(f"I-{dp_tag}")
                
                sent.append(" ")
                label_sequence.append("O")

                labels.extend([f"B-{dp_tag}", f"I-{dp_tag}","O"])

    labels = set(labels)


    return pd.DataFrame.from_dict(sents), labels