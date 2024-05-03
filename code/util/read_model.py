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
def is_nested_numpy_array(data):
    if isinstance(data, list) and all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], np.ndarray) for item in data):
        return True
    return False
def dimensions(lst):
    if not isinstance(lst, list):
        return 0    # 非列表返回0维
    
    if not lst:
        return 1    # 空列表返回1维 
    
    return 1 + dimensions(lst[0]) # 递归调用

class read_model_class():
    def __init__(self,learning_model,save_path,bayes_model) -> None:
        self.learning_model = learning_model
        self.save_path = save_path
        self.bayes_model = bayes_model
        self._build_model()


    def _build_model(self):
        if self.bayes_model in ['atconf']:
            with open(self.save_path+'_scaler.p','rb') as d:
                self.scaler_x,self.scaler_y,self.unimportant_features_id = pickle.load(d)
        elif self.bayes_model in ['robotune','restune','tuneful']:
            with open(self.save_path+'_scaler.p','rb') as d:
                self.scaler_x,self.scaler_y = pickle.load(d)

        if self.learning_model in ["RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror","DECART"]:
            # print(self.save_path)
            try:
                with open(self.save_path+'.p', 'rb') as d:
                    self.model = pickle.load(d)
            except TypeError:
                self.model = pickle.load(open(self.save_path+'.p', 'rb'))
        elif self.learning_model in ["DaL"]:
            self.models = []
            
            with open(self.save_path+'_configs.p', 'rb') as d:
                self.configs = pickle.load(d)
            with open(self.save_path+'_lr_opt.p', 'rb') as d:
                lr_opt = pickle.load(d)
            with open(self.save_path+'_forest.p', 'rb') as d:
                self.forest = pickle.load(d)
            with open(self.save_path+'_dalx.p', 'rb') as d:
                X_train = pickle.load(d)
            with open(self.save_path+'_daly.p', 'rb') as d:
                Y_train = pickle.load(d)      
            if dimensions(Y_train) == 3:
                Y_train = [i[0] for i in Y_train]
            # print(len(X_train), len(Y_train), lr_opt)
            for i in range(len(X_train)):
                model = MLPSparseModel(self.configs[i])
                model.build_train()
                model.train(X_train[i], Y_train[i], lr_opt[i])
                self.models.append(model)  # save the trained models for each cluster
        elif self.learning_model in ["DaL_RF"]:
            self.models = []
            with open(self.save_path+'_forest.p', 'rb') as d:
                self.forest = pickle.load(d)    
            # print(len(X_train), len(Y_train), lr_opt)
            with open(self.save_path+'.p', 'rb') as d:
                self.models = pickle.load(d) 

        elif self.learning_model in["HINNPerf"]:

            with open(self.save_path+'_deps.p', 'rb') as d:
                Y_train = pickle.load(d)
    
            with open(self.save_path+'_indeps.p', 'rb') as d:
                X_train = pickle.load(d)    
           
            with open(self.save_path+'_config.p','rb') as d:
                self.config = pickle.load(d)
            # print(dimensions(Y_train))
            try:
                if dimensions(Y_train.tolist()) == 2:
                    Y_train = [i[0] for i in Y_train]
            except Exception:
                if dimensions(Y_train) == 2:
                    Y_train = [i[0] for i in Y_train]

            data_gen = DataPreproc(X_train, Y_train)
            runner = ModelRunner(data_gen, MLPHierarchicalModel)
            runner.train(self.config)
            self.model = runner
  
        elif self.learning_model in ["DeepPerf"]:
            with open(self.save_path+'_deps.p', 'rb') as d:
                Y_train = pickle.load(d)
                Y_train = [[i] for i in Y_train]
            with open(self.save_path+'_indeps.p', 'rb') as d:
                X_train = pickle.load(d)    
            # config = {'input_dim': 8, 'num_neuron': 128, 'num_block': 3, 'num_layer_pb': 3, 'lamda': 0.001, 'linear': False, 'gnorm': True, 'lr': 0.001, 'decay': None, 'verbose': True}
            with open(self.save_path+'_config.p','rb') as d:
                config = pickle.load(d)
            with open(self.save_path+'_lr_opt.p', 'rb') as d:
                lr_opt = pickle.load(d)
            deepperf_model = MLPSparseModel(config)
            deepperf_model.build_train()
            # print(np.array(X_train), np.array(Y_train), lr_opt)

            # print(dimensions(Y_train))
            
            if dimensions(Y_train) == 3 or is_nested_numpy_array(Y_train)==True:
                Y_train = [i[0] for i in Y_train]
            # print(Y_train)
            deepperf_model.train(np.array(X_train), np.array(Y_train), lr_opt)
            self.model = deepperf_model

        elif self.learning_model in ["Perf_AL"]:
            # PATH = "./Pickle_all/PickleLocker_flash_models/Data_small/Apache_AllNumeric/Perf_AL_seed19_step14"
            with open(self.save_path+'_features_miny_maxy.p','rb') as d:
                [N_features,min_Y,max_Y] = pickle.load(d)
            model = PerfAL_model.MyModel(N_features)
            model = torch.load(self.save_path)
            model.eval()
            self.model = model

        
    def predict(self,solution):
        
        if self.bayes_model in ['atconf']:
            solution = self.scaler_x.transform([solution])
            solution = np.delete(solution, self.unimportant_features_id, axis=1)        

        elif self.bayes_model in ['robotune','restune','tuneful']:
            with open(self.save_path+'_scaler.p','rb') as d:
                self.scaler_x,self.scaler_y = pickle.load(d)
            solution = self.scaler_x.transform([solution])
            # print(model.predict(solution))
        else:
            solution = [solution]
        
        if self.learning_model in ["RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror","DECART"]:
            result_pred = self.model.predict(solution)
        elif self.learning_model in ["DaL"]:
            with open(self.save_path+'_weight.p', 'rb') as d:
                weights = pickle.load(d) 
            with open(self.save_path+'_max_X.p', 'rb') as d:
                max_X = pickle.load(d) 
            with open(self.save_path+'_max_Y.p', 'rb') as d:
                max_Y = pickle.load(d)  
            with open(self.save_path+'_dalx.p', 'rb') as d:
                X_train = pickle.load(d)
            solution = np.divide(solution, max_X)
            for j in range(len(solution)):  ## 预测在哪一个类
                solution[j] = solution[j] * weights[j] 
            temp_cluster = self.forest.predict([solution]) 
            if len(self.configs) == 1 or len(X_train) == 1:
                result_pred = self.models[0].predict(solution)[0]*max_Y
            else:
                for d in range(len(self.configs)): ## 匹配成功聚类之后进行预测
                    if d == temp_cluster:
                        result_pred = self.models[d].predict(solution)[0]*max_Y
        elif self.learning_model in ["DaL_RF"]:
            with open(self.save_path+'_weight.p', 'rb') as d:
                weights = pickle.load(d) 
            with open(self.save_path+'_max_X.p', 'rb') as d:
                max_X = pickle.load(d) 
            with open(self.save_path+'_max_Y.p', 'rb') as d:
                max_Y = pickle.load(d) 
            import copy
            X = copy.deepcopy(solution)

            X = np.array(X[0],dtype=np.float64)
            # print(X)
            # print(weights)
            for j in range(len(X)):  ## 预测在哪一个类
                X[j] = X[j] * weights[j] 
            # print(X)
            temp_cluster = self.forest.predict([X]) 
            # print(temp_cluster)
            # print(len(self.models))
            if len(self.models) == 1:
                result_pred = self.models[0].predict(np.divide(solution, max_X))[0]*max_Y
            else:
                for d in range(len(self.models)): ## 匹配成功聚类之后进行预测
                    if d == temp_cluster[0]:
                        result_pred = self.models[d].predict(np.divide(solution, max_X))[0]*max_Y
    
        elif self.learning_model in["HINNPerf"]:
            result_pred = self.model.predict(solution,self.config)[0][0]
        elif self.learning_model in ["DeepPerf"]:
            result_pred = self.model.predict(solution)[0]
        elif self.learning_model in ["Perf_AL"]:
            with open(self.save_path+'_features_miny_maxy.p','rb') as d:
                [_,min_Y,max_Y] = pickle.load(d)
            x = np.array(solution,dtype=np.float32)
            x = Variable(torch.tensor(x))
            result_pred = self.model(x).detach().numpy()
            result_pred = (float(max_Y-min_Y))*result_pred+float(min_Y)
            result_pred = result_pred[0]


        if self.bayes_model in ['atconf','restune','tuneful','robotune']:
            result_pred = abs(self.scaler_y.inverse_transform(result_pred.reshape(1,-1)))[0]
        if self.bayes_model in ['ottertune']:
            if dimensions(result_pred) == 2:
                result_pred = result_pred[0]

        return result_pred
