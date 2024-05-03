from doepy import build, read_write
from itertools import product
import numpy as np
import pandas as pd
import random
from numpy import genfromtxt
import time
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold

def option_wise_sampling(feature_dict={}):
    final_configs = []
    for i, key in enumerate(feature_dict.keys()):
        temp_config = []
        for j, key2 in enumerate(feature_dict.keys()):
            if i == j:
                temp_config.append(1)
            else:
                temp_config.append(0)
        final_configs.append(temp_config)

    return final_configs

def pair_wise_sampling(feature_dict={}):
    final_configs = []
    for i, key in enumerate(feature_dict.keys()):
        for j, key2 in enumerate(feature_dict.keys()):
            temp_config = []
            for k, key3 in enumerate(feature_dict.keys()):
                if k == j or k == i:
                    temp_config.append(1)
                else:
                    temp_config.append(0)
            final_configs.append(temp_config)

    return final_configs

def Plackett_Burman_design(feature_dict={}, level=2, n=30):
    final_configs = []
    chosen_values = []

    for i, key in enumerate(feature_dict.keys()):
        temp_chosen_values = []
        if level == 1:
            temp_chosen_values.append(np.median(feature_dict[key]))
        elif level == 2:
            temp_chosen_values.append(min(feature_dict[key]))
            temp_chosen_values.append(max(feature_dict[key]))
        elif level > len(feature_dict[key]):
            temp_chosen_values = (feature_dict[key])
        else:
            temp_chosen_values.append(min(feature_dict[key]))
            temp_chosen_values.append(max(feature_dict[key]))
            temp_index = int(len(feature_dict[key]) / (level - 1))
            for k in range(1,int((level - 1) / 2 + 1)):
                temp_chosen_values.append(feature_dict[key][temp_index * k])
            for k in range(1, level - int((level - 1) / 2 + 1)):
                # temp_chosen_values.append(feature_dict[key][temp_index * k])
                temp_chosen_values.append(feature_dict[key][-1 - temp_index * k])
        chosen_values.append(temp_chosen_values)
    # print(chosen_values)

    if np.max(np.max(chosen_values)) > 100:
        for i in (chosen_values):
            final_configs.append(i)
    else:
        for i in product(*chosen_values):
            final_configs.append(i)

    random.seed(1)
    if n < len(final_configs):
        return random.sample(final_configs, n)
    else:
        return final_configs

def random_design(feature_dict={}, n=30):
    final_configs = []
    chosen_values = []
    for i, key in enumerate(feature_dict.keys()):
        chosen_values.append(list(feature_dict[key]))

    for i in (chosen_values):
        final_configs.append(i)

    final_configs = random.sample(final_configs, n)

    random.seed(1)
    if n < len(final_configs):
        return random.sample(final_configs, n)
    else:
        return final_configs

def random_sampling(whole_data, n=30):
    total_index = range(whole_data.shape[0])
    random.seed(1)
    chosen_index = random.sample(total_index, n)
    final_configs = whole_data[chosen_index,:]
    return final_configs

def get_inputs(system_name):
    dir_data = 'Data/{}.csv'.format(system_name)
    # print('Reading ', dir_data)
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)  # read the csv file into array
    (N, n) = whole_data.shape  # get the number of rows and columns
    N_features = n - 1
    N_binary = 0
    N_numeric = 0
    for f in range(N_features):
        if len(set(whole_data[:, f])) <= 2:
            N_binary += 1
        else:
            N_numeric += 1
    # print([N_binary, N_numeric, N])
    return np.array([N_binary, N_numeric, N])

def generate_training_sizes(features, off_feature):

    del features[list(features.keys())[-1]]
    N_features = len(features.keys())

    for temp_i, key in enumerate(features.keys()):
        if off_feature == temp_i:
            features[key] = [np.min(list(set(features[key])))]
        else:
            features[key] = list(set(features[key]))

    binary_features = {}
    numeric_features = {}
    for key in features.keys():
        if max(features[key]) == 1:
            binary_features[key] = (features[key])
        else:
            numeric_features[key] = (features[key])

    # print('Binary features: {}'.format(binary_features))
    # print('Numeric features: {}'.format(numeric_features))

    ow_samples = option_wise_sampling(binary_features)
    # print('Option-wise samples: {}'.format(len(ow_samples)))

    pw_samples = pair_wise_sampling(binary_features)
    # print('Pair-wise samples: {}'.format(len(pw_samples)))

    rb_samples = Plackett_Burman_design(numeric_features, level=7, n=49)
    rb_samples_125 = Plackett_Burman_design(numeric_features, level=5, n=125)
    random_samples = random_design(numeric_features, n=len(numeric_features))
    # print('Random design samples: {}'.format(len(random_samples)))
    # print('Plackett_Burman (49, 7) samples: {}'.format(len(rb_samples)))
    # print('Plackett_Burman (125, 5) samples: {}'.format(len(rb_samples_125)))

    sample_sizes=[len(ow_samples)*len(random_samples), len(ow_samples)*len(rb_samples), len(ow_samples)*len(rb_samples_125), len(pw_samples)*len(random_samples), len(pw_samples)*len(rb_samples), len(pw_samples)*len(rb_samples_125)]
    return sorted(sample_sizes)
