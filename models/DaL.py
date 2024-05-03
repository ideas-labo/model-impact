import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import genfromtxt
from imblearn.over_sampling import SMOTE
import os
from collections import Counter
from doepy import read_write
from random import sample
from utils.general import build_model
from utils.hyperparameter_tuning import nn_l1_val, hyperparameter_tuning
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from utils.mlp_sparse_model_tf2 import MLPSparseModel
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from utils.SPL_sampling import generate_training_sizes
from utils.general import get_non_zero_indexes, process_training_data
import warnings
from util.read_model import read_model_class
from util.get_objective_model import get_path
import pickle
from utils.HINNPerf_args import list_of_param_dicts
from utils.HINNPerf_data_preproc import system_samplesize, seed_generator, DataPreproc
from utils.HINNPerf_args import list_of_param_dicts
from utils.HINNPerf_models import MLPHierarchicalModel
from utils.HINNPerf_model_runner import ModelRunner
warnings.filterwarnings('ignore')

def dimensions(lst):
    if not isinstance(lst, list):
        return 0    # 非列表返回0维
    if not lst:
        return 1    # 空列表返回1维 
    return 1 + dimensions(lst[0]) # 递归调用

def DaL(X_train,Y_train,file_name,seed):
    # start11 = time.time()
    if isinstance(Y_train[0], np.ndarray):
        Y_train = [list(i) for i in Y_train]

    if dimensions(Y_train) == 2:
        Y_train = [i[0] for i in Y_train]


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    
    N_train = len(X_train)
    total_tasks = 1 
    Y_train_else = [[i] for i in Y_train]
    whole_data = np.concatenate([X_train, Y_train_else], axis=1) 

    
    # whole_data = genfromtxt(file_name, delimiter=',', skip_header=1)
    # print(type(whole_data))
    # print(whole_data)
    (N, n) = whole_data.shape
    n = n - 1

    # delete the zero-performance samples
    non_zero_indexes = range(N-1)

    # print('Total sample size: ', len(non_zero_indexes))
    N_features = n + 1 - total_tasks
    # print('N_features: ', N_features)
    test_mode = False ####
    max_depth = 1  ### to run experiments comparing different depths (RQ3), add depths here, starting from 1
    min_samples = 2 
    # compute the weights of each feature using Mutual Information, for eliminating insignificant features
    ## whole_data为whole_data是一个N行n列的数组，表示N个样本的n个特征
    weights = []
    feature_weights = mutual_info_regression(whole_data[non_zero_indexes, 0:N_features],
                                                whole_data[non_zero_indexes, -1], random_state=0)
    # print('Computing weights of {} samples'.format(len(non_zero_indexes)))
    for i in range(N_features):
        weight = feature_weights[i]
        # print('Feature {} weight: {}'.format(i, weight))
        weights.append(weight)
    # print(weights)
    # print('\n---DNN_DaL depth {}---'.format(max_depth))
    # initialize variables
    max_X = []
    max_Y = []
    config = []
    lr_opt = []
    models = []
    X_train = []
    Y_train = []
    X_train1 = []
    Y_train1 = []
    X_train2 = []
    Y_train2 = []
    X_test = []
    Y_test = []
    cluster_indexes_all = []

    # generate clustering labels based on the dividing conditions of DT
    # print('Dividing...')
    # get the training X and Y for clustering
    Y = whole_data[non_zero_indexes, -1][:, np.newaxis]
    X = whole_data[non_zero_indexes, 0:N_features]

    # build and train a CART to extract the dividing conditions
    DT = DecisionTreeRegressor(random_state=seed)
    DT.fit(X, Y)
    tree_ = DT.tree_  # get the tree structure

    # the function to extract the dividing conditions recursively,
    # and divide the training data into clusters (divisions)
    from sklearn.tree import _tree


    def recurse(node, depth, samples=[]):
        indent = "  " * depth
        if depth <= max_depth:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:  # if it's not the leaf node
                left_samples = []
                right_samples = []
                # get the node and the dividing threshold
                name = tree_.feature[node]
                threshold = tree_.threshold[node]
                # split the samples according to the threshold
                for i_sample in range(0, len(samples)):
                    if X[i_sample, name] <= threshold:
                        left_samples.append(samples[i_sample])
                    else:
                        right_samples.append(samples[i_sample])
                # check if the minimum number of samples is statisfied
                if (len(left_samples) <= min_samples or len(right_samples) <= min_samples):
                    print('{}Not enough samples to cluster with {} and {} samples'.format(indent,
                                                                                            len(left_samples),
                                                                                            len(right_samples)))
                    cluster_indexes_all.append(samples)
                else:
                    print("{}{} samples with feature {} <= {}:".format(indent, len(left_samples), name,
                                                                        threshold))
                    recurse(tree_.children_left[node], depth + 1, left_samples)
                    print("{}{} samples with feature {} > {}:".format(indent, len(right_samples), name,
                                                                        threshold))
                    recurse(tree_.children_right[node], depth + 1, right_samples)
        # the base case: add the samples to the cluster
        elif depth == max_depth + 1:
            cluster_indexes_all.append(samples)


    # run the defined recursive function above
    recurse(0, 1, non_zero_indexes)

    k = len(cluster_indexes_all)  # the number of divided subsets

    # if there is only one cluster, DaL can not be used
    if k <= 1:
        print(
            'Error: samples are less than the minimum number (min_samples={}), please add more samples'.format(
                min_samples))
        # return "error"  # end this run

    # extract training samples from each cluster
    N_trains = []  # N_train for each cluster
    cluster_indexes = []
    for i in range(k):
        if int(N_train) > len(cluster_indexes_all[i]):  # if N_train is too big
            N_trains.append(int(len(cluster_indexes_all[i])))
        else:
            N_trains.append(int(N_train))
        # sample N_train samples from the cluster
        cluster_indexes.append(random.sample(cluster_indexes_all[i], N_trains[i]))

    # generate the samples and labels for classification
    total_index = cluster_indexes[0]  # samples in the first cluster
    clusters = np.zeros(int(len(cluster_indexes[0])))  # labels for the first cluster
    for i in range(k):
        if i > 0:  # the samples and labels for each cluster
            total_index = total_index + cluster_indexes[i]
            clusters = np.hstack((clusters, np.ones(int(len(cluster_indexes[i]))) * i))

    # get max_X and max_Y for scaling
    max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)  # scale X to 0-1
    if 0 in max_X:
        max_X[max_X == 0] = 1
    max_Y = np.max(whole_data[total_index, -1]) / 100  # scale Y to 0-100
    if max_Y == 0:
        max_Y = 1

    # get the training data for each cluster
    for i in range(k):  # for each cluster
        temp_X = whole_data[cluster_indexes[i], 0:N_features]
        temp_Y = whole_data[cluster_indexes[i], -1][:, np.newaxis]
        # Scale X and Y
        X_train.append(np.divide(temp_X, max_X))
        Y_train.append(np.divide(temp_Y, max_Y))


    # split train data into 2 parts for hyperparameter tuning
    for i in range(0, k):
        N_cross = int(np.ceil(X_train[i].shape[0] * 2 / 3))
        X_train1.append(X_train[i][0:N_cross, :])
        Y_train1.append(Y_train[i][0:N_cross, :])
        X_train2.append(X_train[i][N_cross:N_trains[i], :])
        Y_train2.append(Y_train[i][N_cross:N_trains[i], :])

    # process the sample to train a classification model
    X_smo = whole_data[total_index, 0:N_features]
    y_smo = clusters
    for j in range(N_features):
        X_smo[:, j] = X_smo[:, j] * weights[j]  # assign the weight for each feature

    # build a random forest classifier to classify testing samples
    forest = RandomForestClassifier(random_state=seed)
    # tune the hyperparameters if not in test mode
    if not test_mode:
        param = {'n_estimators':np.arange(10,100,20),
                 'criterion':['gini']
                    }
        # print('DaL RF Classifier Tuning...')
        gridS = GridSearchCV(forest, param)
        gridS.fit(X_smo, y_smo)
        forest = RandomForestClassifier(**gridS.best_params_, random_state=seed)
    forest.fit(X_smo, y_smo)  # training



    ### Train DNN_DaL
    print('----Training DaL----')
    if test_mode == True:  # default hyperparameters, just for testing
        # for i in range(0, k):
            # define the configuration for constructing the NN
            # temp_lr_opt = 0.001
            # n_layer_opt = 3
            # lambda_f = 0.123
            # temp_config = dict()
            # temp_config['num_neuron'] = 128
            # temp_config['num_input'] = N_features
            # temp_config['num_layer'] = n_layer_opt
            # temp_config['lambda'] = lambda_f
            # temp_config['verbose'] = 0
        save_path = get_path(learning_model="DaL",file=file_name,bayes_models="None",seed=seed)
        # print(save_path)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        with open(save_path+'_configs.p','rb') as d:
            config = pickle.load(d)
    
    ## tune DNN for each cluster (division) with multi-thread
    elif test_mode == False:  # only tune the hyperparameters when not test_mode
        # from concurrent.futures import ThreadPoolExecutor
        
        # # create a multi-thread pool
        # with ThreadPoolExecutor(max_workers=2) as pool:
        #     args = []  # prepare arguments for hyperparameter tuning
        #     for i in range(k):  # for each division
        #         args.append([N_features, X_train1[i], Y_train1[i], X_train2[i], Y_train2[i]])
        #     # optimal_params contains the results from the function 'hyperparameter_tuning'
        #     for optimal_params in pool.map(hyperparameter_tuning, args):
        #         print('Learning division {}... ({} samples)'.format(i + 1, len(X_train[i])))
        #         n_layer_opt, lambda_f, temp_lr_opt = optimal_params  # unzip the optimal parameters
        #         # define the configuration for constructing the DNN
        #         temp_config = dict()
        #         temp_config['num_neuron'] = 128
        #         temp_config['num_input'] = N_features
        #         temp_config['num_layer'] = n_layer_opt
        #         temp_config['lambda'] = lambda_f
        #         temp_config['verbose'] = 0
        #         config.append(temp_config)
        #         lr_opt.append(temp_lr_opt)
        config = dict(
        input_dim = [len(X_train[0][0])],
        num_neuron = [128],
        num_block = [3,4],
        num_layer_pb = [3,4],
        lamda = [0.01,  0.1,  10.],
        linear = [False],
        gnorm = [True],
        lr = [0.001],
        decay = [None],
        verbose = [True]
    )
 
    config_list = list_of_param_dicts(config)
    best_config = config_list[0]
    # print(k)
    for i in range(k):
        # train a local DNN model using the optimal hyperparameters
        # print(config[i])
        X_train[i] = list(X_train[i])
        Y_train[i] = [j[0] for j in Y_train[i]]
        data_gen = DataPreproc(X_train[i], Y_train[i])
        model = ModelRunner(data_gen, MLPHierarchicalModel)
        model.train(best_config)
        # config[i]['num_input'] = N_features
        # model = MLPSparseModel(config[i])
        # model.build_train()
        # # print(X_train[i].shape, Y_train[i].shape)
        # model.train(X_train[i], Y_train[i], lr_opt[i])
        print(model.predict([X_train[i][0]],best_config)*max_Y)
        print(model.predict([X_train[i][1]],best_config)*max_Y)
        models.append(model)  # save the trained models for each cluster
    # print(X_train,Y_train)
    # end11 = time.time()
    # print("-------------",end11-start11)
    return models,forest,weights,k,N_features,config,lr_opt,X_train,Y_train,max_X,max_Y


    # classify the testing samples
    testing_clusters = []  # classification labels for the testing samples
    X = whole_data[testing_index, 0:N_features]
    for j in range(N_features):
        X[:, j] = X[:, j] * weights[j]  # assign the weight for each feature
    for temp_X in X:
        temp_cluster = forest.predict(temp_X.reshape(1, -1))  # predict the dedicated local DNN using RF
        testing_clusters.append(int(temp_cluster))
    # print('Testing size: ', len(testing_clusters))
    # print('Testing sample clusters: {}'.format((testing_clusters)))
    Y_pred_test = []
    for d in range(k):
        if d == testing_clusters[i]:
            Y_pred_test.append(max_Y * models[d].predict(X_test[i][np.newaxis, :]))