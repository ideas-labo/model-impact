# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from scipy import stats
from numpy import genfromtxt
import os
import random
from collections import Counter
from random import sample
from doepy import read_write
from sklearn.metrics import mean_squared_error
from utils.SPL_sampling import generate_training_sizes
from utils.general import get_non_zero_indexes, process_training_data
from utils.PerfAL_model import *
from utils.PerfAL_util import *
from utils.HINNPerf_args import list_of_param_dicts


def Perf_AL(X_train,Y_train,dir_data,seed = 2):
    lens = int(2/3*len(X_train))
    Y_train = [[i] for i in Y_train]
    X_train1 = np.array(X_train[:lens],dtype=np.float32)
    X_train2 = np.array(X_train[lens:],dtype=np.float32)
    Y_train1 = np.array(Y_train[:lens],dtype=np.float32)
    Y_train2 = np.array(Y_train[lens:],dtype=np.float32)
    if set([tuple(i) for i in X_train1.tolist()]) == 1:  
        X_train1 = np.array(X_train[:],dtype=np.float32)
        Y_train1 = np.array(Y_train[:],dtype=np.float32)
    # seed = 0
    torch.manual_seed( seed )
    use_gpu = False
    torch.set_default_tensor_type('torch.FloatTensor')
    # config = dict(
    #     max_epoch = [50],
    #     BATCH_SIZE = [64],
    #     C1 = [1],
    #     C2 = [0],
    #     weight_decay = [1e-5,  1e-3,  1,10],
    #     lr_model = [1e-5,1e-3,1e-1,10],
    #     lr_D = [5e-5,5e-3,5e-1,50]
    # )
    # config_list = list_of_param_dicts(config)
    # for config in config_list:
        
    max_epoch = 50
    BATCH_SIZE = 64
    C1, C2, weight_decay = 1,0,1
    lr_model, lr_D = 0.1,5e-3


    # lambda1 = 0.5
    

    

    # total_tasks = 1
    # print('Dataset: ' + dir_data)
    # whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    # (N, n) = whole_data.shape
    # n = n - 1

    # # delete the zero-performance samples
    # non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)

    # print('Total sample size: ', len(non_zero_indexes))
    # N_features = n + 1 - total_tasks
    # print('N_features: ', N_features)
    N_features = len(X_train1[1])
    # print('N_features: ', N_features)
    # N_experiments = 30
    # start = 0
    # print('N_expriments: ', N_experiments)
    # print(X_train1, Y_train1)
    train_set = MyDataset(X_train1, Y_train1)
    train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True )

    valid_set = MyDataset(X_train2, Y_train2)
    valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=True )

    # test_set = MyDataset(X_train2,Y_train2)
    # test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=True )

    model = MyModel(N_features)
    criterion_MSE = nn.MSELoss()

    if use_gpu:
        model.cuda()
        criterion_MSE.cuda()


    optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr_model, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=5 )  # 学习率衰减

    model_D = Model_D()
    criterion_BCE = nn.BCELoss()

    if use_gpu:
        model_D.cuda()
        criterion_BCE.cuda()

    optimizer_D = optim.Adam( model_D.parameters(), lr=lr_D, weight_decay=weight_decay )

    def l1_loss(var):
        return torch.abs(var).sum()


    print( '-------------- Training Perf_AL --------------' )

    for epoch in range( max_epoch ):

        # print( 'Epoch %d' % epoch )
        t0 = time.time()

        values = [epoch]
        ################ Training on Train Set ################

        y_train = np.zeros( 0, dtype=float )
        pred_train = np.zeros( 0, dtype=float )

        for i, data in enumerate( train_loader, 0 ):
            # print(data)
            # ------------ Prapare Variables ------------
            batch_x, batch_y = data

            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            if use_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            real_labels = sample_real_labels( batch_x.size(0), dir_data)
            #real_labels = batch_y
            real_labels = Variable(real_labels)

            label_one = Variable( torch.ones( batch_x.size(0) ) )      # 1 for real
            label_zero = Variable( torch.zeros( batch_x.size(0) ) )    # 0 for fake

            if use_gpu:
                real_labels = real_labels.cuda()
                label_one = label_one.cuda()
                label_zero = label_zero.cuda()

            # ------------ Training model_D ------------
            optimizer_D.zero_grad()
            model.eval()  # lock model

            output_real = model_D( real_labels )
            if (len(output_real) == 1):
                loss_real = criterion_BCE(output_real.squeeze().unsqueeze(0), label_one)
            else:
                loss_real = criterion_BCE( output_real.squeeze(), label_one )
            acc_real = ( output_real >= 0.5 ).data.float().mean()
            # print(X_train,Y_train)
            # print(batch_x.squeeze())
            fake_labels = model( batch_x.squeeze() ).detach()
            # print(fake_labels)
            # print(type(batch_x.squeeze()))
            output_fake = model_D( fake_labels )
            if (len(output_fake) == 1):
                loss_fake = criterion_BCE(output_fake.squeeze().unsqueeze(0), label_zero)
            else:
                # print(output_fake.squeeze(), label_zero)
                loss_fake = criterion_BCE( output_fake.squeeze(), label_zero )
            # loss_fake = criterion_BCE( output_fake.squeeze(), label_zero )
            acc_fake = ( output_fake < 0.5 ).data.float().mean()


            loss_D = loss_real + loss_fake
            acc_D = ( acc_real + acc_fake ) / 2

            loss_D.backward()
            optimizer_D.step()

            # ------------ Training model ------------
            model.train()  # unlock model
            optimizer.zero_grad()


            fake_labels = model( batch_x )
            output = model_D( fake_labels )

            fake_labels=fake_labels.float()
            batch_y=batch_y.float()
            if (len(output) == 1):
                term2 = C2 * criterion_BCE(output.squeeze().unsqueeze(0), label_one)
            else:
                term2 = C2 * criterion_BCE( output.squeeze(), label_one )
            term1 = C1 * criterion_MSE( fake_labels, batch_y )
            # term2 = C2 * criterion_BCE( output.squeeze(), label_one )
            l1_regular = float(torch.tensor(0))
            l2_regular = float(torch.tensor(0))
            for param in model.parameters():
                l1_regular += torch.norm(param, 1)
                l2_regular += torch.norm(param, 2)


        #     l1_regular = lambda1 * l1_loss(fake_labels)

            loss = term1 + term2
            #loss = term1 + term2 + l1_regular
            loss.backward()
            optimizer.step()

            # ------------ Preparation for Evaluation on Train Set ------------
            fake_labels = fake_labels.cpu().data.numpy() if use_gpu else fake_labels.data.numpy()
            batch_y = batch_y.cpu().data.numpy() if use_gpu else batch_y.data.numpy()


        ################ Evaluation on Train Set ################

        # mse = mean_squared_error( batch_y, fake_labels )
        # rank_spearman = stats.spearmanr( get_rank(batch_y), get_rank(fake_labels) )[0]
        # values.append( mse )
        # values.append( rank_spearman )
        # print( 'Train Set\tmse=%f, rank_spearman=%f' % ( mse, rank_spearman ) )

        ################ Evaluation on Valid/Test Set ################

        model.eval()  # evaluation mode
        
        mre, mse, rank_spearman = model_eval( model, valid_loader, use_gpu )
        # values.append( mre )
        # values.append( mse )
        # values.append( rank_spearman )
        # print( 'Valid Set\tmre=%f, mse=%f, rank_spearman=%f' % ( mre, mse, rank_spearman ) )


        scheduler.step( rank_spearman )
        
        # if epoch == max_epoch-1:
            # mre, mse, rank_spearman = test_eval( model, test_loader, use_gpu )
            # Y_pred_test = get_predicted_values( model, test_loader, use_gpu )
            # if np.min(Y_pred_test) >= 10:
            #     Y_pred_test = np.around(Y_pred_test, decimals=2)
        # else:
        #     mre, mse, rank_spearman = model_eval( model, test_loader, use_gpu )
        # values.append( mre )
        # values.append( mse )
        # values.append( rank_spearman )
        # print( 'Test Set\tmre=%f, mse=%f, rank_spearman=%f' % ( mre, mse, rank_spearman ) )
    x0 = np.array([X_train[0]],dtype=np.float32)
    x0 = Variable(torch.tensor(x0))
    x1 = np.array([X_train[5]],dtype=np.float32)
    x1 = Variable(torch.tensor(x1))
    x2 = np.array([X_train[3]],dtype=np.float32)
    x2 = Variable(torch.tensor(x1))
    a = model(x0).detach().numpy()[0][0]
    b = model(x1).detach().numpy()[0][0]
    c = model(x2).detach().numpy()[0][0]
    # if a!=1 and a!=0 and b!=1 and b!=0 and c!=1 and c!=0:
        # print(config)
    print(a)
    print(b)
    print(c)
    return model,N_features
