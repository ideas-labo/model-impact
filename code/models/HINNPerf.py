import argparse
import numpy as np
import time
import os
import random
from random import sample
from doepy import read_write
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from utils.SPL_sampling import generate_training_sizes
from utils.general import get_non_zero_indexes, process_training_data
from utils.HINNPerf_data_preproc import system_samplesize, seed_generator, DataPreproc
from utils.HINNPerf_args import list_of_param_dicts
from utils.HINNPerf_models import MLPHierarchicalModel
from utils.HINNPerf_model_runner import ModelRunner
import warnings
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


def HINNPerf(X_train,Y_train,file_name):

    # print(Y_train)
    # 转化为合适格式
    if isinstance(Y_train[0], np.ndarray):
        Y_train = [list(i) for i in Y_train]
    if dimensions(Y_train) == 2:
        Y_train = [i[0] for i in Y_train]
    test_mode = False ####
    # print(X_train,Y_train)

    data_gen = DataPreproc(X_train,Y_train)
    runner = ModelRunner(data_gen, MLPHierarchicalModel)


    config = dict(
        input_dim = [len(X_train[0])],
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
    print('----Training HINNPerf----')
    abs_error_val_min = float('inf')
    best_config = None
    if test_mode:
        best_config = config_list[0]
        # save_path = get_path(learning_model="HINNPerf",file=file_name)
        # print(save_path)
        # with open(save_path+'_config.p','rb') as d:
        #     config = pickle.load(d)
        # print("best config: ",best_config)
        runner.train(best_config)
    else:
        for con in config_list:
            # print(con)
            abs_error_train, abs_error_val = runner.train(con)
            # print(abs_error_val)
            if abs_error_val_min > abs_error_val:
                abs_error_val_min = abs_error_val
                best_config = con
        runner.train(best_config)
    print("best config: ",best_config)
    print(runner.predict([X_train[0]],best_config))
    print(runner.predict([X_train[1]],best_config))
    return runner, best_config
