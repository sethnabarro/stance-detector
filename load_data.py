# coding=utf-8
import csv
import pandas as pd

_TRAIN_DATA_PATH = 'data/SemEval2016-Task6-subtaskA-traindata-gold.csv'
_TEST_DATA_PATH = 'data/SemEval2016-Task6-subtaskA-testdata-gold.txt'


def load_train_data(csv_path=_TRAIN_DATA_PATH):
    """
    Read in training data. Adapted from 'stance detection data processing.html'
    :param csv_path: filepath to training csv
    :type csv_path: str
    :return training data table
    :type pandas DataFrame
    """
    data = []

    with open(csv_path, 'r',
              encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, quotechar='"')
        columns = next(reader)
        for line in reader:
            data.append(line)

    train_df = pd.DataFrame(data, columns=columns)

    return train_df


def load_test_data(txt_path=_TEST_DATA_PATH):
    """
    Read in test data. Adapted from 'stance detection data processing.html'
    :param txt_path: filepath to test txt file
    :type txt_path: str
    :return test data table
    :type pandas DataFrame
    """
    data = []

    with open(txt_path, 'r',
              encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, delimiter='\t')
        columns = next(reader)
        for line in reader:
            data.append(line)

    test_df = pd.DataFrame(data, columns=columns)

    return test_df
