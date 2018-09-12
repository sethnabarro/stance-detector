# coding=utf-8
"""An Ordinal Logistic Regression model for stance classification"""
import mord

import stance_detect_base

# Regularisation paramater
_PARAM_GRID = {'alpha': [10. ** x for x in range(-3, 3)]}

# mord.LogisticIT has very stringent tolerance criteria for convergence
# Need to restrict maximum iterations
_MAX_ITER = 200


class StanceDetectorLogitOrd(stance_detect_base.StanceDetector):
    def __init__(self, name='Ordinal Logistic Regression'):
        super(StanceDetectorLogitOrd, self).__init__()
        self.model_sk = mord.LogisticIT(max_iter=_MAX_ITER)
        self.name = name

    def train(self, X_tr, y_tr, cv_param_grid=_PARAM_GRID):
        super(StanceDetectorLogitOrd, self).train(X_tr, y_tr, _PARAM_GRID)
