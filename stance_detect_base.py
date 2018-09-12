# coding=utf-8
"""A stance classification skeleton"""
import copy
import numpy as np
import pickle
from sklearn import base
from sklearn import model_selection
from sklearn import metrics

import utils

_NUM_FOLDS = 5


class StanceDetector(object):
    """Parent class for Stance Detection"""
    def __init__(self):
        super(StanceDetector, self).__init__()
        self._model_sk = None

    @property
    def model_sk(self):
        return self._model_sk

    @model_sk.setter
    def model_sk(self, model_to_set):
        self._model_sk = model_to_set

    def cross_validation(self, X_tr, y_tr, cv_param_grid):
        """Given search space of parameters to test, to cross validation to find optimum"""
        _gs = model_selection.GridSearchCV(
            self._model_sk, param_grid=cv_param_grid, scoring=self._cv_scorer(), cv=_NUM_FOLDS)
        _gs.fit(X_tr, y_tr)
        self._model_sk = _gs.best_estimator_

    @staticmethod
    def _cv_scorer():
        """Returns the function for calculating F1 score (macro-averaged over for/against)
        as the required type for evaluation during cross-validation"""
        return metrics.make_scorer(utils.f1_for_against)

    def train(self, X_tr, y_tr, cv_param_grid=None):
        """
        Train the model on the given data.
        :param X_tr: numpy array
        :param y_tr: numpy array, single dimension
        :param cv_param_grid: dict: param name -> list of values to try. If given cross validation
        carried out
        """
        # Encode training labels, str -> int
        y_tr_enc = np.vectorize(utils.LABEL_ENCODER.get)(y_tr)
        if cv_param_grid is not None:
            self.cross_validation(X_tr, y_tr_enc, cv_param_grid)
        self._model_sk.fit(X_tr, y_tr_enc)

    def predict(self, X_te):
        preds_raw = self._model_sk.predict(X_te)

        # Decode predictions int -> str (for, none or against)
        return np.vectorize(utils.LABEL_DECODER.get)(preds_raw)

    def f1_score(self, X_te, y_te):
        """F1 score on test data, macro averaged over for/against classes"""
        preds = self.predict(X_te)

        return utils.f1_for_against(y_te, preds)

    def confusion_matrix(self, X_te, y_te):
        preds = self.predict(X_te)
        return utils.confusion_matrix_df(y_te, preds)

    def clone(self):
        clone = copy.deepcopy(self)

        # Clone the sklearn model
        clone.model_sk = base.clone(self.model_sk)
        return clone

    def predict_proba(self, X_te):
        """
        Predict probabilities of class membership
        :param X_te: numpy array with same dimension as X_tr used for training
        :return: X_te.shape[0] X 3 numpy array, predicted probabilities of being in each class
        """
        proba_preds = self._model_sk.predict_proba(X_te)

        # Reorder so AGAINST, NONE, FOR
        return proba_preds[:, np.argsort(self._model_sk.classes_)]
