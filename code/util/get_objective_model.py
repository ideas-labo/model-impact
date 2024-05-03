from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import pickle
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from utils.HINNPerf_data_preproc import DataPreproc
from utils.HINNPerf_model_runner import ModelRunner
from utils.HINNPerf_models import MLPHierarchicalModel
from utils.mlp_sparse_model_tf2 import MLPSparseModel
import torch
from torch.autograd import Variable
import re
import os



# def get_objective_score_with_model(solution,learning_model,file):
#     solution = [solution]
#     save_path = 

def get_path(learning_model,file,bayes_models,seed):
    # print(learning_model,file,bayes_models,seed)
    pattern = re.compile(r'{0}(_\w+)'.format(learning_model)) 
    pattern_num = re.compile(r'step(\d+).')
    pattern_seed = re.compile(r'step(\d+).')
    root = './Pickle_all/PickleLocker_'+str(bayes_models)+'_models/'+str(file[:-4])+"/"
    names = os.listdir(root)
    names_tmp = []

    for name in names:

        m = re.search(r"_seed(\d+)", name)
        if m:
            if int(m.group(1)) == int(seed): 
                names_tmp.append(name)
    names = names_tmp
    
    

    def sort_key(filename):
        match = pattern_num.search(filename)
        if match:
            return int(match.group(1))
        else:
            return 0
    f_names = []

    for name in names:
        match = pattern.search(name)
        if match:
            if learning_model == "RF":
                pattern1 = re.compile(r'{0}(_\w+)'.format("DaL"))
                match1 = pattern1.search(name)
                if not match1:
                    f_names.append(root+name)
            else:    
                f_names.append(root+name)



    new_strings = [re.sub(r'(step\d+)_\w+.p', r'\1', s) for s in f_names]
    new_strings = [re.sub(r'(step\d+)+.p', r'\1', s) for s in new_strings]
    f_names = list(set(new_strings))

    f_names.sort(key=sort_key)
    # print(f_names)
    save_path = f_names[0]
    
    return save_path

def get_objective_score_with_model(solution,learning_model,file):
    solution = [solution]
    pattern = re.compile(r'{0}(_\w+)'.format(learning_model)) 
    pattern_num = re.compile(r'step(\d+).')
    root = './Pickle_all/PickleLocker_None_models/'+str(file[:-4])+"/"
    names = os.listdir(root)
    def sort_key(filename):
            match = pattern_num.search(filename)
            if match:
                return int(match.group(1))
            else:
                return 0
    f_names = []

    for name in names:
        match = pattern.search(name)
        if match:
            f_names.append(root+name)



    new_strings = [re.sub(r'(step\d+)_\w+.p', r'\1', s) for s in f_names]
    new_strings = [re.sub(r'(step\d+)+.p', r'\1', s) for s in new_strings]
    f_names = list(set(new_strings))

    f_names.sort(key=sort_key)
    save_path = f_names[4]

    return read_model_free(solution,learning_model,save_path)[0]




def read_model_free(solution,learning_model,save_path):
    if learning_model in ["RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror","DECART"]:
        with open(save_path+'.p', 'rb') as d:
            model = pickle.load(d)
            result_pred = model.predict(solution)
    elif learning_model in ["DaL"]:
        models = []
        
        with open(save_path+'_configs.p', 'rb') as d:
            configs = pickle.load(d)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        with open(save_path+'_forest.p', 'rb') as d:
            forest = pickle.load(d)
        with open(save_path+'_dalx.p', 'rb') as d:
            X_train = pickle.load(d)
        with open(save_path+'_daly.p', 'rb') as d:
            Y_train = pickle.load(d)      
        with open(save_path+'_weight.p', 'rb') as d:
            weights = pickle.load(d) 
        with open(save_path+'_max_X.p', 'rb') as d:
            max_X = pickle.load(d) 
        with open(save_path+'_max_Y.p', 'rb') as d:
            max_Y = pickle.load(d)     
        for i in range(len(configs)):
            model = MLPSparseModel(configs[i])
            model.build_train()
            model.train(X_train[i], Y_train[i], lr_opt[i])
            models.append(model)  # save the trained models for each cluster
     
        solution = np.divide(solution, max_X)
        
        for j in range(len(solution)):  ## 预测在哪一个类
            solution[j] = solution[j] * weights[j] 
        temp_cluster = forest.predict(solution) 
        if len(configs) == 1:
            result_pred = models[d].predict(solution)[0]*max_Y
        else:
            for d in range(len(configs)): ## 匹配成功聚类之后进行预测
                if d == temp_cluster:
                    result_pred = models[d].predict(solution)[0]*max_Y
        return result_pred
    elif learning_model in["HINNPerf"]:

        with open(save_path+'_deps.p', 'rb') as d:
            Y_train = pickle.load(d)
  
        with open(save_path+'_indeps.p', 'rb') as d:
            X_train = pickle.load(d)    
        # print(np.array(X_train), np.array(Y_train))
        # config = {'input_dim': 8, 'num_neuron': 128, 'num_block': 3, 'num_layer_pb': 3, 'lamda': 0.001, 'linear': False, 'gnorm': True, 'lr': 0.001, 'decay': None, 'verbose': True}
        with open(save_path+'_config.p','rb') as d:
            config = pickle.load(d)
        data_gen = DataPreproc(X_train, Y_train)
        runner = ModelRunner(data_gen, MLPHierarchicalModel)
        runner.train(config)
        result_pred = runner.predict(solution,config)[0][0]

    elif learning_model in ["DeepPerf"]:
        with open(save_path+'_deps.p', 'rb') as d:
            Y_train = pickle.load(d)
            Y_train = [[i] for i in Y_train]
        with open(save_path+'_indeps.p', 'rb') as d:
            X_train = pickle.load(d)    
        # config = {'input_dim': 8, 'num_neuron': 128, 'num_block': 3, 'num_layer_pb': 3, 'lamda': 0.001, 'linear': False, 'gnorm': True, 'lr': 0.001, 'decay': None, 'verbose': True}
        with open(save_path+'_config.p','rb') as d:
            config = pickle.load(d)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        deepperf_model = MLPSparseModel(config)
        deepperf_model.build_train()
        # print(np.array(X_train), np.array(Y_train), lr_opt)
        deepperf_model.train(np.array(X_train), np.array(Y_train), lr_opt)
        result_pred = deepperf_model.predict(solution)[0]
    

    elif learning_model in ["Perf_AL"]:
        # PATH = "./Pickle_all/PickleLocker_flash_models/Data_small/Apache_AllNumeric/Perf_AL_seed19_step14"
        with open(save_path+'_features_miny_maxy.p','rb') as d:
            [N_features,min_Y,max_Y] = pickle.load(d)
        model = PerfAL_model.MyModel(N_features)
        model = torch.load(save_path)
        model.eval()
        x = np.array(solution,dtype=np.float32)
        x = Variable(torch.tensor(x))
        result_pred = model(x).detach().numpy()
        result_pred = (float(max_Y-min_Y))*result_pred+float(min_Y)
        result_pred = result_pred[0]

    return result_pred



def read_model(solution,learning_model,bayes_model,dataset,seed=1,step=6):
    
    save_path = './Pickle_all/PickleLocker_'+str(bayes_model)+'_models/Data_small/'+str(dataset)+'/'+str(learning_model)+'_seed'+str(seed)+'_step'+str(step)

    if bayes_model in ['atconf']:
        with open(save_path+'_scaler.p','rb') as d:
            scaler_x,scaler_y,unimportant_features_id = pickle.load(d)
        solution = scaler_x.transform([solution])
        solution = np.delete(solution, unimportant_features_id, axis=1)        
    elif bayes_model in ['robotune','restune','tuneful']:
        with open(save_path+'_scaler.p','rb') as d:
            scaler_x,scaler_y = pickle.load(d)
        solution = scaler_x.transform([solution])
        # print(model.predict(solution))
    else:
        solution = [solution]


    if learning_model in ["RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror","DECART"]:
        with open(save_path+'.p', 'rb') as d:
            model = pickle.load(d)
            result_pred = model.predict(solution)
    elif learning_model in ["DaL"]:
        models = []
        
        with open(save_path+'_configs.p', 'rb') as d:
            configs = pickle.load(d)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        with open(save_path+'_forest.p', 'rb') as d:
            forest = pickle.load(d)
        with open(save_path+'_dalx.p', 'rb') as d:
            X_train = pickle.load(d)
        with open(save_path+'_daly.p', 'rb') as d:
            Y_train = pickle.load(d)      
        with open(save_path+'_weight.p', 'rb') as d:
            weights = pickle.load(d) 
        with open(save_path+'_max_X.p', 'rb') as d:
            max_X = pickle.load(d) 
        with open(save_path+'_max_Y.p', 'rb') as d:
            max_Y = pickle.load(d)     
        for i in range(len(configs)):
            model = MLPSparseModel(configs[i])
            model.build_train()
            model.train(X_train[i], Y_train[i], lr_opt[i])
            models.append(model)  # save the trained models for each cluster
     
        solution = np.divide(solution, max_X)
        
        for j in range(len(solution)):  ## 预测在哪一个类
            solution[j] = solution[j] * weights[j] 
        temp_cluster = forest.predict(solution) 
        if len(configs) == 1:
            result_pred = models[d].predict(solution)[0]*max_Y
        else:
            for d in range(len(configs)): ## 匹配成功聚类之后进行预测
                if d == temp_cluster:
                    result_pred = models[d].predict(solution)[0]*max_Y
        return result_pred
    elif learning_model in["HINNPerf"]:

        with open(save_path+'_deps.p', 'rb') as d:
            Y_train = pickle.load(d)
  
        with open(save_path+'_indeps.p', 'rb') as d:
            X_train = pickle.load(d)    
        # print(np.array(X_train), np.array(Y_train))
        # config = {'input_dim': 8, 'num_neuron': 128, 'num_block': 3, 'num_layer_pb': 3, 'lamda': 0.001, 'linear': False, 'gnorm': True, 'lr': 0.001, 'decay': None, 'verbose': True}
        with open(save_path+'_config.p','rb') as d:
            config = pickle.load(d)
        data_gen = DataPreproc(X_train, Y_train)
        runner = ModelRunner(data_gen, MLPHierarchicalModel)
        runner.train(config)
        result_pred = runner.predict(solution,config)[0][0]

    elif learning_model in ["DeepPerf"]:
        with open(save_path+'_deps.p', 'rb') as d:
            Y_train = pickle.load(d)
            Y_train = [[i] for i in Y_train]
        with open(save_path+'_indeps.p', 'rb') as d:
            X_train = pickle.load(d)    
        # config = {'input_dim': 8, 'num_neuron': 128, 'num_block': 3, 'num_layer_pb': 3, 'lamda': 0.001, 'linear': False, 'gnorm': True, 'lr': 0.001, 'decay': None, 'verbose': True}
        with open(save_path+'_config.p','rb') as d:
            config = pickle.load(d)
        with open(save_path+'_lr_opt.p', 'rb') as d:
            lr_opt = pickle.load(d)
        deepperf_model = MLPSparseModel(config)
        deepperf_model.build_train()
        # print(np.array(X_train), np.array(Y_train), lr_opt)
        deepperf_model.train(np.array(X_train), np.array(Y_train), lr_opt)
        result_pred = deepperf_model.predict(solution)[0]
    

    # elif learning_model in ["Perf_AL"]:
    #     # PATH = "./Pickle_all/PickleLocker_flash_models/Data_small/Apache_AllNumeric/Perf_AL_seed19_step14"
    #     with open(save_path+'_features_miny_maxy.p','rb') as d:
    #         [N_features,min_Y,max_Y] = pickle.load(d)
    #     model = PerfAL_model.MyModel(N_features)
    #     model = torch.load(save_path)
    #     model.eval()
    #     x = np.array(solution,dtype=np.float32)
    #     x = Variable(torch.tensor(x))
    #     result_pred = model(x).detach().numpy()
    #     result_pred = (float(max_Y-min_Y))*result_pred+float(min_Y)
    #     result_pred = result_pred[0]

    
    if bayes_model in ['atconf','restune','tuneful','robotune']:
        result_pred = abs(scaler_y.inverse_transform(result_pred.reshape(1,-1)))[0]
    if bayes_model in ['ottertune']:
        result_pred = result_pred[0]

    return result_pred
    
# print(get_objective_score_with_model([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],"HINNPerf"))
