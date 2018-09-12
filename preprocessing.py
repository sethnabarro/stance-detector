# coding=utf-8
import nltk
import os
import string

try:
    if 'stopwords' not in os.listdir(nltk.data.find('corpora')):
        # Have downloaded nltk corpora before (so folder exists - error not raised),
        # but no stopwords
        nltk.download('stopwords')
    if 'vader_lexicon.zip' not in os.listdir(nltk.data.find('sentiment')):
        # Lexicon for sentiment analyser
        nltk.download('vader_lexicon')
except IndexError:
    # No corpora downloaded yet, error raised when nltk checks non-existent folder
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

_RELEVANT_PUNCT = set('!?$@%#')   # Characters which may be informative of stance


def tokenise_series(a_series):
    """
    Tokenise the string in each element of a series
    :param a_series: to tokenise
    :type: pandas Series, each element of type str
    :return pandas Series, each element is a list
    """
    tweet_tok = nltk.tokenize.TweetTokenizer()
    return a_series.apply(tweet_tok.tokenize)


def stem_series(a_series):
    """
    Stem all tokens in given series of tokenised documents
    :param a_series: Series of tokenised strings
    :type: pandas Series, each element is a list of str
    :return: stemmed inputs
    :type: pandas Series, each element list of strings
    """
    stem_porter = nltk.stem.PorterStemmer()
    return a_series.apply(lambda token_list: list(map(stem_porter.stem, token_list)))


def remove_stop_words(a_series, stop_words=()):
    """
    Remove all stopwords in given series of documents
    :param a_series: many tokenised documents
    :type a_series: pandas Series of lists of strings
    :param stop_words: collection of words to remove
    :type stop_words: list or tuple
    :return: tokenised documents, without stopwords
    :type: pandas Series of lists of strings
    """
    if not any(stop_words):
        # Load english stopwords from NLTK
        stop_words = nltk.corpus.stopwords.words('english')

    def _remove_stop_words(token_list):
        return list(filter(lambda tok: tok.lower() not in stop_words, token_list))

    return a_series.apply(_remove_stop_words)


def remove_irrelevant_punctuation(tweet_series):
    """
    Method to delete punctuation thought irrelevant to stance from tweets.
    NOT IN USE - found to reduce classification accuracy
    :param tweet_series: series of tweets
    :type tweet_series: pandas Series, each element str or unicode
    :return: series of tweets with irrelevant punctuation removed
    """
    irrel_punc = set(string.punctuation).difference(_RELEVANT_PUNCT)
    translator = str.maketrans({_p: None for _p in irrel_punc})

    return tweet_series.str.translate(translator)


def preprocess_tweets(tweet_series):
    """
    Takes series of raw tweets and tokenises, stems, removes stop words
    :param tweet_series: Tweets to preprocess
    :type tweet_series: pandas Series, each element is str
    :return: tweet_series after preprocessing
    :type: pandas Series, each element is list of strings
    """
    # Take out irrelevant punctuation
    # NOT USED - found to degrade classification accuracy
    # processed = remove_irrelevant_punctuation(tweet_series)

    # Tokenise
    processed = tokenise_series(tweet_series)

    # Remove stop words
    processed = remove_stop_words(processed)

    # Stem
    processed = stem_series(processed)

    return processed
