# coding=utf-8
import numpy as np
import random

import task_a
import utils

# Set seed
np.random.seed(8)
random.seed(8)

_SAVE_MODEL = False


def main():
    best_mod, best_mod_feats = task_a.model_selection_and_evaluation()
    if _SAVE_MODEL:
        utils.save_model(best_mod, 'best_model.pkl')


if __name__ == '__main__':
    main()
