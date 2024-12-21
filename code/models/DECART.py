import numpy as np
import pandas as pd
import warnings
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from random import sample
from utils.SPL_sampling import generate_training_sizes
from utils.general import get_non_zero_indexes, process_training_data
from collections import Counter
from doepy import read_write
warnings.filterwarnings('ignore')


def DECART(X_train,Y_train):
    test_mode = False
    model = DecisionTreeRegressor(random_state=0)
    # DECART is DT with resampling and hyperparameter tuning.
    max = 5
    if len(X_train) > 5:
        max = 10
    param = {'criterion': ('squared_error', 'friedman_mse'),
                'splitter': ('best', 'random'),
                'min_samples_split': np.arange(2, max, 1)
                }
    if not test_mode:
        # print('Hyperparameter Tuning...')
        gridS = GridSearchCV(model, param)
        gridS.fit(X_train, Y_train)
        model = DecisionTreeRegressor(**gridS.best_params_, random_state=0)

    model.fit(X_train, Y_train)
    return model