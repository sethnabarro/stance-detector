# coding=utf-8
import csv
import pandas as pd

_TRAIN_DATA_PATH = 'data/SemEval2016-Task6-subtaskA-traindata-gold.csv'
_TEST_DATA_PATH = 'data/SemEval2016-Task6-subtaskA-testdata-gold.txt'


def load_train_data(csv_path=_TRAIN_DATA_PATH):
    """
    Read in training data.
    :param csv_path: filepath to training csv
    :type csv_path: str
    :return training data table
    :type pandas DataFrame
    """
    train_df = pd.read_csv(open(csv_path, 'r', encoding="iso-8859-1"), quotechar='"')

    return train_df


def load_test_data(txt_path=_TEST_DATA_PATH):
    """
    Read in test data
    :param txt_path: filepath to test txt file
    :type txt_path: str
    :return test data table
    :type pandas DataFrame
    """
    test_df = pd.read_csv(open(txt_path, 'r', encoding="iso-8859-1"), delimiter='\t')

    return test_df
