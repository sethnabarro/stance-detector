# coding=utf-8
import dill
import numpy as np
import pandas as pd
from sklearn import metrics as metrics_sk

LABEL_ENCODER = {'AGAINST': -1, 'NONE': 0, 'FAVOR': 1}
LABEL_DECODER = {_int: _txt for _txt, _int in LABEL_ENCODER.items()}
SENTIMENT_ENCODER = {'neg': 'AGAINST', 'other': 'NONE', 'pos': 'FAVOR'}
STANCE_CLASSES = ['AGAINST', 'NONE', 'FAVOR']

# How many n-grams to use as features in bag-of-ngrams representation
_MAX_NGRAM_FEATS = 4000

# Arguments for creatig bag-of-words/bag-of-grams repr
BOW_KWARGS = {'per_target': False, 'normalise_counts': False,
              'stop_words': 'english', 'ngram_range': (1, 2), 'max_feats': _MAX_NGRAM_FEATS}

# Number of decimal places to save latex tables to
NUM_DP = 2


def save_model(model, model_fn):
    """Save model to file using dill"""
    with open(model_fn, 'wb') as _model_file:
        dill.dump(model, _model_file)


def load_model(model_fn):
    """Load serialised model from file"""
    with open(model_fn, 'rb') as _model_file:
        loaded_model = dill.load(_model_file)
    return loaded_model


def f1_for_against(y_truth, y_pred):
    """
    F1 score, macro-averaged over FAVOR/AGAINST classes
    :param y_truth: true classes
    :param y_pred: predicted classes
    :return: F1 score
    """
    # Get order of labels
    if y_truth.dtype == np.int64:
        label_order = [-1, 1]
    elif y_truth.dtype == '<U7':
        label_order = ['AGAINST', 'FAVOR']

    # Classes with no predicted labels (happens during cross-val)
    # cause ill-defined div by 0 in f1 score
    # Calculate for one class at a time, testing if labels present
    if np.any(np.isin(label_order[0], y_pred)):
        f1_against = metrics_sk.f1_score(
            y_truth, y_pred, labels=[label_order[0]], average='macro')
    else:
        f1_against = 0.
    if np.any(np.isin(label_order[1], y_pred)):
        f1_for = metrics_sk.f1_score(
            y_truth, y_pred, labels=[label_order[1]], average='macro')
    else:
        f1_for = 0.

    # Macro average of for/against f1
    return (f1_for + f1_against) * 0.5


def split_train_validation(df_tr, tr_frac=0.7):
    """
    Split train set into reduced train and validation. By randomly sampling tr_frac
    within each target class
    :param df_tr: pandas DataFrame
    :param tr_frac: float, fraction to assign to reduced train
    :return: tuple of dataframes: train, validation
    """
    # Initialise empty dataframe
    df_tr_tr = pd.DataFrame(columns=df_tr.columns)
    df_tr_val = pd.DataFrame(columns=df_tr.columns)

    # Now sample same fraction from each target class, append to dfs to return
    for _, _df in df_tr.groupby('Target'):
        tr_idx = np.random.choice(_df.index, replace=False, size=round(_df.shape[0] * tr_frac))
        df_tr_tr = df_tr_tr.append(_df.loc[tr_idx])
        df_tr_val = df_tr_val.append(_df.drop(tr_idx))

    # Sort by index
    df_tr_tr.sort_index(inplace=True)
    df_tr_val.sort_index(inplace=True)

    return df_tr_tr, df_tr_val


def confusion_matrix_df(y_true, y_pred):
    """Wrapper for sklearn confusion matrix, converts to readable df"""
    if y_true.dtype is np.int64:
        label_order = [-1, 0, 1]
    else:
        label_order = ['AGAINST', 'NONE', 'FAVOR']
    conf_mat_np = metrics_sk.confusion_matrix(y_true, y_pred, labels=label_order)
    colnames = [['Predicted'] * len(label_order), label_order]
    idxnames = [['True'] * len(label_order), label_order]
    return pd.DataFrame(conf_mat_np, columns=colnames, index=idxnames)
