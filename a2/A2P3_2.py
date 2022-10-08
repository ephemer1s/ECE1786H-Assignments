import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

if not os.path.exists('./data/train.tsv'):
    if not os.path.exists('./data'):
        os.mkdir('./data')

    try:
        tsv_file = open('/content/drive/MyDrive/ECE1786/data.tsv')
    except:
        tsv_file = open('/data/data.tsv')
    finally:
        read_tsv = csv.reader(tsv_file, delimiter="\t")

    data = pd.DataFrame(data=[row for row in read_tsv], columns=['text', 'label'])
    data.drop([0], axis=0, inplace=True)

    train_set, test_set = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    train_set, val_set = train_test_split(train_set, test_size=0.2, shuffle=True, random_state=42)
    overfit_set = data.sample(n=50)

    train_set.to_csv('./data/train.tsv', sep="\t")
    test_set.to_csv('./data/test.tsv', sep="\t")
    val_set.to_csv('./data/validation.tsv', sep="\t")
    overfit_set.to_csv('./data/overfit.tsv', sep="\t")