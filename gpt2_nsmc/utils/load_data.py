import urllib.request
import pandas as pd


def load_data():

    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

    train_data = pd.read_table("ratings_train.txt")
    test_data = pd.read_table('ratings_test.txt')

    train_data = train_data.dropna(how = 'any')
    train_data = train_data.reset_index(drop=True)

    test_data = test_data.dropna(how = 'any')
    test_data = test_data.reset_index(drop=True)

    return train_data, test_data



