from datasets import load_dataset

def load_data():
    dataset = load_dataset('klue', 'ynat')
    train_test_dataset = dataset['train']
    valid_dataset = dataset['validation']
    train_test = train_test_dataset.train_test_split(train_size=0.95, shuffle=False)
    train_dataset, test_dataset = train_test['train'], train_test['test']

    return train_dataset, valid_dataset ,valid_dataset