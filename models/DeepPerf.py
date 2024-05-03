import numpy as np
from numpy import genfromtxt
import time
import random
from random import sample
from utils.SPL_sampling import generate_training_sizes
from utils.general import get_non_zero_indexes, process_training_data
from utils.hyperparameter_tuning import hyperparameter_tuning
import pandas as pd
import warnings
from utils.mlp_sparse_model_tf2 import MLPSparseModel
from utils.mlp_plain_model_tf2 import MLPPlainModel
from doepy import read_write
import os
from collections import Counter
import csv

from util.read_model import read_model_class
from util.get_objective_model import get_path
import pickle

warnings.filterwarnings('ignore')

def dimensions(lst):
    if not isinstance(lst, list):
        return 0    # 非列表返回0维
    
    if not lst:
        return 1    # 空列表返回1维 
    
    return 1 + dimensions(lst[0]) # 递归调用

def DeepPerf(X_train,Y_train,file_name,seed):
 
    lens = int(2/3*len(X_train))
    if isinstance(Y_train[0], np.ndarray):
        Y_train = [list(i) for i in Y_train]
  
    if dimensions(Y_train) == 1:
        Y_train = [[i] for i in Y_train]
   
    # print(Y_train)
    X_train1 = np.array(X_train[:lens])
    X_train2 = np.array(X_train[lens:])
    Y_train1 = np.array(Y_train[:lens])
    Y_train2 = np.array(Y_train[lens:])
    test_mode = True
    total_tasks = 1
    N_features = len(X_train[0])
    print('---Training DeepPerf---')
    # default hyperparameters, just for testing
    if test_mode == True:
        
        # lr_opt = 0.0001
        # n_layer_opt = 8
        # lambda_f = 4.64
        # config = dict()
        # config['num_neuron'] = 128
        # config['num_input'] = N_features
        # config['num_layer'] = n_layer_opt
        # config['lambda'] = lambda_f
        # config['verbose'] = 0
        save_path = get_path(learning_model="DeepPerf",file=file_name,bayes_models="None",seed=seed)
        with open(save_path+'_config.p','rb') as d:
            config = pickle.load(d)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        config['num_input'] = N_features
            
    # if not test_mode, tune the hyperparameters
    else:
        # print([N_features, X_train1, Y_train1, X_train2, Y_train2])
        n_layer_opt, lambda_f, lr_opt = hyperparameter_tuning(
            [N_features, X_train1, Y_train1, X_train2, Y_train2])

        # save the hyperparameters
        config = dict()
        config['num_neuron'] = 128
        config['num_input'] = N_features
        config['num_layer'] = n_layer_opt
        config['lambda'] = lambda_f
        config['verbose'] = 0
        print(n_layer_opt, lambda_f, lr_opt)
    # train the DeepPerf model
    deepperf_model = MLPSparseModel(config)
    deepperf_model.build_train()
    deepperf_model.train(np.array(X_train), np.array(Y_train), lr_opt)
    print(deepperf_model.predict([X_train[0]]))
    print(deepperf_model.predict([X_train[1]]))
    return deepperf_model,config,lr_opt
