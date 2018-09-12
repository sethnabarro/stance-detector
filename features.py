# coding=utf-8
"""Functions for calculation of features"""
from nltk import sentiment, tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text as text_sk
from sklearn import preprocessing as preprocessing_sk

import preprocessing


def bag_of_words(tr_tweets, te_tweets, tr_targets=pd.Series(), te_targets=pd.Series(),
                 per_target=False, max_feats=None, normalise_counts=False, **kwargs):
    """
    Calculate bag-of-words representations of train and test tweets
    :param tr_tweets: pandas Series of strings, raw texts to convert (from train set)
    :param te_tweets: pandas Series of strings, raw texts to convert (from test set)
    :param tr_targets: pandas Series of strings, target classes (from train set)
    :param te_targets: pandas Series of strings, target classes (from test set)
    :param per_target: bool, whether to find separate BoW repr for each target class
    :param max_feats: int, maximum number of words/ngrams to keep, number of dimensions
    in returned feature matrices
    :param normalise_counts: bool, whether to divide the counts within each tweet by the
    number of tokens (not for Multinomial NB)
    :param kwargs: to be passed onto sklearn CountVectorizer
    :return: tuple, training feature matrix, test feature matrix, list of feature names
    (with '_bow' appended to each)
    """

    if per_target and not tr_targets.empty and not te_targets.empty:
        # Create different BoW for each target
        # Only useful if using max_features - as most common words/n-grams
        # May be for only one or two of the targets
        x_tr = np.zeros((tr_tweets.shape[0], max_feats), dtype=np.int64)
        x_te = np.zeros((te_tweets.shape[0], max_feats), dtype=np.int64)
        for _targ in tr_targets.unique():
            word_bagger = text_sk.CountVectorizer(max_features=max_feats, **kwargs)
            x_tr[(tr_targets == _targ).values] = \
                word_bagger.fit_transform(tr_tweets[(tr_targets == _targ).values].values).toarray()
            x_te[(te_targets == _targ).values] = \
                word_bagger.transform(te_tweets[(te_targets == _targ).values].values).toarray()
    else:
        word_bagger = text_sk.CountVectorizer(max_features=max_feats, **kwargs)
        x_tr = word_bagger.fit_transform(tr_tweets).toarray()
        x_te = word_bagger.transform(te_tweets).toarray()

    if normalise_counts:
        # Normliase counts by length of tweet
        tr_tweet_lens = tr_tweets.apply(tokenize.TweetTokenizer().tokenize).apply(len)
        te_tweet_lens = te_tweets.apply(tokenize.TweetTokenizer().tokenize).apply(len)
        x_tr = np.divide(x_tr, tr_tweet_lens.values[:, np.newaxis])
        x_te = np.divide(x_te, te_tweet_lens.values[:, np.newaxis])
    return x_tr, x_te, [_fn + '_bow' for _fn in word_bagger.get_feature_names()]


def hand_crafted_features(tweets, get_names=False):
    """
    Calculate some additional features from raw tweets
    :param tweets: pandas Series of str, raw tweets
    :param get_names: bool, whether to return list of feature names with feature array
    :return: tuple(numpy array, list(str)), or numpy array
    """
    funcs_to_apply = [('tweet_len', len),                         # Number of chars of tweet
                      ('!_count', lambda _str: _str.count('!')),  # Number of exclamation marks
                      ('?_count', lambda _str: _str.count('?')),  # Number of question marks
                      ('%_upper',                                 # % of tweet capitalised
                       lambda _str: sum(1. for _char in _str if _char.isupper()) / len(_str))]

    feats = []
    for _, _func in funcs_to_apply:
        feats.append(tweets.apply(_func))

    if get_names:
        # Append '_hc' to each feature name to indicate 'hand-crafted'
        return np.array(feats).T, [_n + '_hc' for _n, _ in funcs_to_apply]
    else:
        return np.array(feats).T


def infer_sentiment(tweets, get_names=False):
    """
    Estimate the sentiment based on lexical rule-based VADER model
    See https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109
    """
    sia = sentiment.vader.SentimentIntensityAnalyzer()
    sentiment_dicts = tweets.apply(sia.polarity_scores).tolist()
    if get_names:
        return pd.DataFrame(sentiment_dicts)[['neg', 'pos']].values, ['neg_sa', 'pos_sa']
    else:
        return pd.DataFrame(sentiment_dicts)[['neg', 'pos']].values


def calculate_features(tr_tweets, te_tweets, targets_tr, targets_te, use_tfidf=False,
                       w_sentiment=True, w_handcrafted=True, **bow_kwargs):
    """
    Calculate all features, combine together into two arrays: one for tr and one for te
    :param tr_tweets: pandas Series of strings, raw texts to convert (from train set)
    :param te_tweets: pandas Series of strings, raw texts to convert (from test set)
    :param targets_tr: pandas Series of strings, target classes (from train set)
    :param targets_te: pandas Series of strings, target classes (from test set)
    :param use_tfidf: bool, whether to convert BoW to TF-IDF
    :param w_sentiment: bool, whether to include sentiment analysis inferences as features
    :param w_handcrafted: bool, whether to include handcrafted features
    :return: tuple: numpy array of training data, numpy array of test data, list of feature names
    """

    # Preprocess tweets (tokenise, stem, remove stop words)
    tr_tokens = preprocessing.preprocess_tweets(tr_tweets)
    te_tokens = preprocessing.preprocess_tweets(te_tweets)

    # Now join preprocessed tokenised tweets into single strings,
    # necessary for input to CountVectorizer
    tr_tweet_proc = tr_tokens.apply(lambda _tokens: ' '.join(_tokens))
    te_tweet_proc = te_tokens.apply(lambda _tokens: ' '.join(_tokens))

    # Calculate bag-of-words representation
    x_tr, x_te, feature_names = bag_of_words(
        tr_tweet_proc, te_tweet_proc, targets_tr, targets_te, **bow_kwargs)

    if use_tfidf:
        # Convert BoW to TF-IDF
        tfidfer = text_sk.TfidfTransformer()
        x_tr = tfidfer.fit_transform(x_tr)
        x_te = tfidfer.transform(x_te)

    if w_handcrafted:
        # Add handcrafted features
        x_tr_hc, feature_names_hc = hand_crafted_features(tr_tweets, get_names=True)
        x_tr = np.hstack((x_tr, x_tr_hc))
        x_te = np.hstack((x_te, hand_crafted_features(te_tweets)))
        feature_names.extend(feature_names_hc)

    if w_sentiment:
        # Add inferred sentiment features
        x_tr_sent, sent_feat_names = infer_sentiment(tr_tweets, get_names=True)
        x_tr = np.hstack((x_tr, infer_sentiment(tr_tweets)))
        x_te = np.hstack((x_te, infer_sentiment(te_tweets)))
        feature_names.extend(sent_feat_names)
    return x_tr, x_te, feature_names


def scale_inputs(x_tr, x_te):
    """
    Zero-mean, unit-variance scaling, fitted to training inputs,
    applied to both training and test
    :param x_tr: Training inputs
    :type x_tr: numpy array
    :param x_te: Test inputs, same dimension as x_te
    :type x_te: numpy array
    :return: scaled training and test inputs
    :type: tuple of numpy arrays
    """
    ss_x = preprocessing_sk.StandardScaler()
    x_tr = ss_x.fit_transform(x_tr)
    x_te = ss_x.transform(x_te)
    return x_tr, x_te
