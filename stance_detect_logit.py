# coding=utf-8
"""A Logistic Regression model for stance classification"""
from sklearn.linear_model import logistic

import stance_detect_base


# Regularisation paramater
_PARAM_GRID = {'C': [10. ** x for x in range(-3, 4)]}


class StanceDetectorLogit(stance_detect_base.StanceDetector):
    def __init__(self, name='Logistic Regression'):
        super(StanceDetectorLogit, self).__init__()
        self.model_sk = logistic.LogisticRegression()
        self.name = name

    def train(self, X_tr, y_tr, cv_param_grid=_PARAM_GRID):
        super(StanceDetectorLogit, self).train(X_tr, y_tr, _PARAM_GRID)

    def get_coef(self):
        return self.model_sk.coef_

    def get_intercept(self):
        return self.model_sk.intercept_

    def get_classes(self):
        return self.model_sk.classes_
