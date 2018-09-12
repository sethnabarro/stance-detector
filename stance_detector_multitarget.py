# coding=utf-8
"""Stance classification model which trains a submodel for each target"""
import matplotlib.pyplot as plt
import numpy as np

import stance_detect_logit as logit
import utils


class StanceDetectorMultiTarget(object):
    def __init__(self, stance_detector, name='multi-target stance detector'):
        self._stance_detector = stance_detector
        self._models_per_target = {}
        self.name = name

    @staticmethod
    def _data_per_target(targets, *iter_of_arrays):
        """Generator of arrays, filters each of iter_of_arrays into subarray for each target"""
        for _targ in np.unique(targets):
            yield _targ, (_arr[targets == _targ] for _arr in iter_of_arrays)

    def train(self, X_tr, y_tr, targets_tr):
        """Train a model for each target class"""
        for _targ, (_X_tr_targ, _y_tr_targ) in self._data_per_target(targets_tr, X_tr, y_tr):
            self._models_per_target[_targ] = self._stance_detector.clone()
            self._models_per_target[_targ].train(_X_tr_targ, _y_tr_targ)

    def predict(self, X_te, targets_te):
        """Prediction using separate model for each target class"""
        predictions = np.empty((X_te.shape[0],), dtype='<U7')  # Unicodes
        for _targ, (_X_te_targ,) in self._data_per_target(targets_te, X_te):
            predictions[targets_te == _targ] = self._models_per_target[_targ].predict(_X_te_targ)
        return predictions

    def predict_proba(self, X_te, targets_te):
        """Predict probabilities of class membership using separate model for each target class"""
        num_classes = len(self._models_per_target[targets_te[0]].model_sk.classes_)
        predictions_prob = np.empty((X_te.shape[0], num_classes), dtype=float)
        for _targ, (_X_te_targ,) in self._data_per_target(targets_te, X_te):
            predictions_prob[targets_te == _targ] = self._models_per_target[_targ].predict_proba(_X_te_targ)
        return predictions_prob

    def f1_score(self, X_te, y_te, targets_te, per_target=False):
        """F1 score, macro-averaged over for/against classess
        :param per_target: bool, if true return dict of F1 scores, one for each class."""
        if per_target:
            f1_scores_per_target = {}
            for _targ, (_X_te_targ, _y_te_targ) in self._data_per_target(targets_te, X_te, y_te):
                f1_scores_per_target[_targ] = \
                    self._models_per_target[_targ].model_f1_score(_X_te_targ, _y_te_targ)
            return f1_scores_per_target
        else:
            preds_all_targets = self.predict(X_te, targets_te)

            f1_all = utils.f1_for_against(y_te, preds_all_targets)
            return f1_all

    def confusion_matrix(self, X_te, y_te, targets_te, per_target=False):
        """
        Confusion matrix calculation: ijth element is number of instances
        predicted to be in jth class but were actually in ith
        """
        if per_target:
            # Get the confusion matrix for each target,
            # return dict: target name -> confusion matrix
            conf_mat_per_target = {}
            for _targ, (_X_te_targ, _y_te_targ) in self._data_per_target(targets_te, X_te, y_te):
                conf_mat_per_target[_targ] = \
                    self._models_per_target[_targ].confusion_matrix(_X_te_targ, _y_te_targ)
            return conf_mat_per_target
        else:
            preds_all_targets = self.predict(X_te, targets_te)
            conf_mat_all = utils.confusion_matrix_df(y_te, preds_all_targets)
            return conf_mat_all

    def get_biggest_feature_weight_ids(self, num_weights):
        """Finds IDs of features, order by feature weights, largest first"""
        # Check all submodels are logistic regression models
        if all([isinstance(_submod, logit.StanceDetectorLogit) for _submod
                in self._models_per_target.values()]):
            feat_ids = {}
            for _targ, _submod in self._models_per_target.items():
                # Get the abs weights for all features, summed over classes
                _submod_weights = np.sum(np.abs(_submod.get_coef()), axis=0)

                # Sort descending and retain first 'num_weights'
                _weights = sorted(enumerate(_submod_weights), key=lambda _c: -_c[1])[:num_weights]

                # Get feat ids
                _top_ids = [_id for _id, _ in _weights]
                feat_ids[_targ] = _top_ids
            return feat_ids
        else:
            raise TypeError('Getting feature weights of submodels only'
                            'supported for Logistic Regression')

    def plot_weights(self, feature_names, num_weights=30, plot_fn=''):
        """Plots the weights of the logit submodel"""
        # Check all submodels are logistic regression models
        if all([isinstance(_submod, logit.StanceDetectorLogit) for _submod
                in self._models_per_target.values()]):
            # Iterate over target topics, creating plot of submodel weights for each
            for _targ, _submod in self._models_per_target.items():
                # Get the abs weights for all features, summed over classes
                _submod_weights = np.sum(np.abs(_submod.get_coef()), axis=0)

                # Sort descending and retain first 'num_weights'
                _weights = sorted(enumerate(_submod_weights), key=lambda _c: -_c[1])[:num_weights]

                # Convert feature ids to names and make barplot
                plt.bar([feature_names[_w[0]] for _w in _weights], [_w[1] for _w in _weights])
                plt.xticks(rotation=90)

                if plot_fn:
                    _plot_fn_targ = '{}_{}.{}'.format(
                        plot_fn.split('.')[0], _targ.lower().split(' ')[0],
                        plot_fn.split('.')[-1])
                    plt.savefig(_plot_fn_targ, bbox_inches='tight')
                    plt.close('all')
                else:
                    plt.title(_targ)
                    plt.show()

        else:
            raise TypeError('Plotting weights of submodels only supported for Logistic Regression')

    def get_intercepts(self):
        """Gets the logistic regression intercepts of each logit submodel"""
        # Check all submodels are logistic regression models
        if all([isinstance(_submod, logit.StanceDetectorLogit) for _submod
                in self._models_per_target.values()]):
            iceps = {}   # To store intercepts for each target
            for _targ, _submod in self._models_per_target.items():
                _icep = _submod.get_intercept()

                # So that intercepts of each class always in same order - sort by class
                _classes = _submod.get_classes()
                iceps[_targ] = _icep[np.argsort(_classes)]
            return iceps
        else:
            raise TypeError('Retrieval of intercepts of submodels only'
                            'supported for Logistic Regression')
