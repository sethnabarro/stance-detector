# coding=utf-8
"""A Na\"{i}ve Bayes model for stance classification"""
import numpy as np
from sklearn import naive_bayes

import stance_detect_base

# Level of smoothing
_PARAM_GRID = {'alpha': np.linspace(1e-10, 3., 10)}


class StanceDetectorNB(stance_detect_base.StanceDetector):
    def __init__(self, name='Naive Bayes'):
        super(StanceDetectorNB, self).__init__()
        self.model_sk = naive_bayes.MultinomialNB()    # fit_prior set to True by default
        self.name = name

    def train(self, X_tr, y_tr, cv_param_grid=_PARAM_GRID):
        super(StanceDetectorNB, self).train(X_tr, y_tr, _PARAM_GRID)
