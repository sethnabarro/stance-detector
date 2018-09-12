# coding=utf-8
import os
import pandas as pd

import features
import load_data
import stance_detect_logit as logit
import stance_detect_logit_ord as logit_ord
import stance_detector_multitarget as multitarget
import stance_detect_nb as nb
import utils

if not os.path.exists('latex/'):
    # Create directory to store tables and figs for report
    os.mkdir('latex/')

if not os.path.exists('latex/tables/'):
    # Create directory to store latex tables
    os.mkdir('latex/tables/')


def assess_model_accuracies(df_tr, df_te, models_and_feats, feature_scaling=False,
                            get_conf_mats=False):
    """
    Given some models, train and test data, evaluate the performance of each
    :param df_tr: training data
    :param df_te: test data
    :param models_and_feats: models with train() and predict method(), and feature sets
    they require. Feature sets is a list of strings, each is a suffix indicating a feature group
    in feat_names below
    :type models_and_feats: dict, Model -> str
    :param feature_scaling: whether to do zero-mean, unit variance scaling on feats (not for NB!)
    :type feature_scaling: bool
    :param get_conf_mats: whether to return confusion matrices for test predictions
    :return: either dict: model name -> f1 scores or
    tuple: (dict: model name -> f1, dict: model_name -> confusion matrix)
    """
    # Get outputs
    y_tr = df_tr['Stance'].values.astype('U')   # Unicode more robust
    y_te = df_te['Stance'].values.astype('U')

    # ...And targets
    targets_tr = df_tr['Target']
    targets_te = df_te['Target']

    # ...And calculate features
    x_tr, x_te, feat_names = \
        features.calculate_features(
            df_tr['Tweet'], df_te['Tweet'], targets_tr, targets_te,
            **utils.BOW_KWARGS)

    if feature_scaling:
        # Zero-mean, unit-variance scaling
        x_tr, x_te = features.scale_inputs(x_tr, x_te)

    # Dict to store results in
    model_f1s = {}
    if get_conf_mats:
        model_cms = {}

    for _model, _feat_sets in models_and_feats.items():
        print('Evaluating performance for model: {}'.format(_model.name))

        # Get ids of features to use
        feat_ids = [_i for _i, _fn in enumerate(feat_names) if _fn.split('_')[-1] in _feat_sets]

        # Collate arguments for training and held-out f1 evaluation
        _train_args = [x_tr.copy()[:, feat_ids], y_tr.copy()]
        _eval_args = [x_te.copy()[:, feat_ids], y_te.copy()]

        # If training separate model for each target - add targets to args
        if isinstance(_model, multitarget.StanceDetectorMultiTarget):
            _train_args.append(targets_tr.values)
            _eval_args.append(targets_te.values)

        # Do training
        _model.train(*_train_args)

        # Assess accuracy
        _f1 = _model.f1_score(*_eval_args)
        _conf_mat = _model.confusion_matrix(*_eval_args)

        # Print metrics
        print('F1 score: {}'.format(_f1))
        print('Confusion matrix:\n{}\n\n'.format(_conf_mat))

        # Store accuracy
        model_f1s[_model.name] = _f1
        if get_conf_mats:
            model_cms[_model.name] = _conf_mat
    if get_conf_mats:
        return model_f1s, model_cms
    else:
        return model_f1s


def model_selection(df_tr, df_val, results_tex_fn='latex/tables/validation_accs.tex'):
    """
    Instantiate candidate models, evaluate accuracies on validation set,
    return highest-scoring
    :param df_tr: Training data table
    :type df_tr: pandas DataFrame
    :param df_val: Validation data table
    :type df_val: pandas DataFrame
    :param results_tex_fn: file path to store latex table of results at
    :type results_tex_fn: str
    :return: The best model and the corresponding feature set as tuple
    """
    # Create instances of all models to test
    lr = logit.StanceDetectorLogit()
    nbayes = nb.StanceDetectorNB()
    lr_ord = logit_ord.StanceDetectorLogitOrd()
    lr_mt = multitarget.StanceDetectorMultiTarget(
        lr.clone(), 'Per-Target Logistic Regression')
    lr_ord_mt = multitarget.StanceDetectorMultiTarget(
        lr_ord.clone(), 'Per-Target Ordinal Logistic Regression')
    nb_mt = multitarget.StanceDetectorMultiTarget(
        nbayes.clone(), 'Per-Target Naive Bayes')

    # Different models use slightly different features. Multinomial NB
    # should use bag-of-words only (other features not multinomial-distributed)
    # Suffixes 'bow', 'hc', 'sa' indicate bag-of-words, hand-crafted and sentiment
    # analysis respectively
    all_feat_types = ['bow', 'hc', 'sa']
    nb_feat_types = ['bow']
    models = {lr: all_feat_types, nbayes: nb_feat_types, lr_ord: all_feat_types,
              lr_mt: all_feat_types, lr_ord_mt: all_feat_types, nb_mt: nb_feat_types}

    # Copy models in current state, such that best performing type of model
    # can be returned in untrained form
    models_fresh = models.copy()
    name_to_model = {_m.name: _m for _m, _ in models_fresh.items()}

    # Get accuracies
    validation_accs = assess_model_accuracies(df_tr, df_val, models)

    # Reformat accs into dataframe - save to latex
    acc_df = pd.DataFrame({'F1 Score': validation_accs})
    acc_df.index.name = 'Model'
    acc_df.round(utils.NUM_DP).reset_index().to_latex(results_tex_fn, index=False)

    # Return model with highest f1
    best_model = name_to_model[max(validation_accs, key=validation_accs.get)]
    best_model_feats = models[best_model]
    return name_to_model[max(validation_accs, key=validation_accs.get)], best_model_feats


def test_set_evaluation(df_tr, df_te, model_and_feats, latex_table_dir='latex/tables/'):
    """
    Calculate test-set metrics for final model trained on full dataset
    :param df_tr: training data
    :param df_te: test data
    :param model_and_feats: dict: model -> feature set
    :param latex_table_dir: str, where to save results
    """
    model_name = list(model_and_feats.keys())[0].name
    f1s, conf_mats = assess_model_accuracies(df_tr, df_te, model_and_feats, get_conf_mats=True)
    f1_table = pd.DataFrame({'F1 Score': [f1s[model_name]]})
    f1_table.round(utils.NUM_DP).to_latex('{}/f1_test.tex'.format(latex_table_dir), index=False)
    conf_mats[model_name].to_latex('{}/confusion_matrix_test.tex'.format(latex_table_dir))


def model_selection_and_evaluation():
    """
    Test some candidate models with validation set, select highest scoring,
    train on full train + validation set, evluate on test set
    :return: tuple: best model, list of feature sets it uses
    """
    # Load train and test sets
    df_tr, df_te = load_data.load_train_data(), load_data.load_test_data()

    # Split train into validation (for model selection) and train
    df_tr_tr, df_tr_val = utils.split_train_validation(df_tr)

    # Assess accuracies of all models on validation set
    # Get best scoring canditate
    best_model, best_model_feats = model_selection(df_tr_tr, df_tr_val)

    print('Best scoring model is: {}, Using feature sets: {}'
          .format(best_model.name, best_model_feats))

    # Evaluate test set accuracy of chosen model
    test_set_evaluation(df_tr, df_te, {best_model: best_model_feats})

    return best_model, best_model_feats
