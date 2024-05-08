
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from torch.autograd import Variable
import torch
import copy
import pickle
import numpy as np
from models import DaL, DECART, DeepPerf, HINNPerf, DaL_RF
from sklearn.pipeline import make_pipeline
import tensorflow.compat.v1 as tf

## bagging方法对所有的预测model
class bagging():
    def __init__(self,train_x,train_y,learning_model,file_name,seed,step,funcname) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.learning_model = learning_model
        self.file_name = file_name
        self.models = []
        self.full_model = None
        self.seed = seed
        self.max_Y = None
        self.min_Y = None
        self.step = step
        self.funcname = funcname

        self.modelss = []  #Dal 
        self.config = None
        self.dalx = None
        self.daly = None
        self.lr_opt = None
        self.forest = None  #Dal
        self.weights = None  #Dal
        self.k = None
        self.N_features = None
        self.max_Y = None
        self.configs_DAL = []
        self.ks_DAL = []

        self.best_configs = [] #HINNPerf
        self.full_config = None

        self.usegpu =  True #Perf-AL
        self.AL_features = None

    def bagging_fit(self,estimators_num = 6):
        lens = len(self.train_y)
        self.models = []
        if self.learning_model in ["DT","RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror"]:
            if self.learning_model in ["DT"]:
                model = RandomForestRegressor(random_state=self.seed)
                model.fit(self.train_x,self.train_y)
                self.full_model = model
            for i in range(estimators_num+1):
                if self.learning_model == 'DT':
                    break
                if self.learning_model == 'LR':
                    model = LinearRegression()
                elif self.learning_model == 'RF':
                    model = DecisionTreeRegressor(random_state=self.seed)
                elif self.learning_model == 'KNN':
                    model = KNeighborsRegressor(n_neighbors=5)
                elif self.learning_model == 'SVR':
                    model = SVR()
                elif self.learning_model == 'KRR':
                    model = KernelRidge()
                elif self.learning_model == 'GP':
                    model = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=self.seed)
                elif self.learning_model == 'SPLConqueror':
                    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
                    # model = SGDRegressor(max_iter=1000, random_state=self.seed)

                if i == estimators_num:
                    try:
                        model.fit(self.train_x,self.train_y)
                    except Exception as e:
                        if self.learning_model == "GP":
                            model = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(), random_state=self.seed)
                            model.fit(self.train_x,self.train_y)
                    self.full_model = model
                    break   
                np.random.seed(self.seed * i)
                bagging_index = np.random.choice(lens,lens,replace=True)
                if len(set(bagging_index)) == 1:
                    bagging_index = np.random.choice(lens,lens-1,replace=False)
                bagging_x = [self.train_x[j] for j in bagging_index]
                bagging_y = [self.train_y[j] for j in bagging_index]
                try:
                    model.fit(bagging_x, bagging_y)
                except Exception as e:
                    if self.learning_model == "GP":
                        model = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(), random_state=self.seed)
                        model.fit(bagging_x, bagging_y)
                self.models.append(model)
        else:

            if self.learning_model == "DaL":
                for i in range(estimators_num+1):
                    if i == estimators_num:
                        self.full_model,self.forest,self.weights,self.k,self.N_features,self.config,self.lr_opt,self.dalx,self.daly,self.max_X,self.max_Y = DaL.DaL(self.train_x,self.train_y,self.file_name,self.seed)
                        break
                    np.random.seed(self.seed * i)
                    bagging_index = np.random.choice(lens,lens,replace=True)
                    if len(set(bagging_index)) == 1:
                        bagging_index = np.random.choice(lens,lens-1,replace=False)
                    bagging_x = [self.train_x[j] for j in bagging_index]
                    bagging_y = [self.train_y[j] for j in bagging_index]
                    models,self.forest_DAL,_,k_DAL,self.N_features,config_DAL,_,_,_,_,_ = DaL.DaL(bagging_x,bagging_y,self.file_name,self.seed)
                    self.ks_DAL.append(k_DAL)
                    self.configs_DAL.append(config_DAL)
                    self.modelss.append(models)
            
            elif self.learning_model == "DaL_RF":
                for i in range(estimators_num+1):
                    if i == estimators_num:
                        self.full_model,self.forest,self.weights,self.N_features,self.k,self.max_X,self.max_Y= DaL_RF.DaL_RF(self.train_x,self.train_y,self.file_name,self.seed)
                        break
                    np.random.seed(self.seed * i)
                    bagging_index = np.random.choice(lens,lens,replace=True)
                    if len(set(bagging_index)) == 1:
                        bagging_index = np.random.choice(lens,lens-1,replace=False)
                    bagging_x = [self.train_x[j] for j in bagging_index]
                    bagging_y = [self.train_y[j] for j in bagging_index]
                    models,self.forest_DAL,_,_,_,_,_ = DaL_RF.DaL_RF(bagging_x,bagging_y,self.file_name,self.seed)
                    self.modelss.append(models)                

            elif self.learning_model == "DECART":
                for i in range(estimators_num+1):
                    if i == estimators_num:
                        self.full_model = DECART.DECART(self.train_x,self.train_y)
                        break
                    np.random.seed(self.seed * i)
                    bagging_index = np.random.choice(lens,lens,replace=True)
                    if len(set(bagging_index)) == 1:
                        bagging_index = np.random.choice(lens,lens-1,replace=False)
                    bagging_x = [self.train_x[j] for j in bagging_index]
                    bagging_y = [self.train_y[j] for j in bagging_index]
                    model = DECART.DECART(bagging_x,bagging_y)
                    self.models.append(model)
            
            elif self.learning_model == "DeepPerf":
                for i in range(estimators_num+1):
                    if i == estimators_num:
                        self.full_model,self.full_config,self.lr_opt = DeepPerf.DeepPerf(self.train_x,self.train_y,self.file_name,self.seed)
                        # print(self.full_model.predict([[0,1,1,1,0,1,0,100,8294400,0.9,8]]))
                        # print(self.full_model.predict([[1,0,0,1,0,0,0,100,8294400,0.9,100]]))
                        break
                    np.random.seed(self.seed * i)
                    bagging_index = np.random.choice(lens,lens,replace=True)
                    # print(bagging_index)
                    while len(set(bagging_index)) == 1:
                        bagging_index = np.random.choice(lens,lens-1,replace=False)
                    bagging_x = [self.train_x[j] for j in bagging_index]
                    bagging_y = [self.train_y[j] for j in bagging_index]
                    model,_,_ = DeepPerf.DeepPerf(bagging_x,bagging_y,self.file_name,self.seed)
                    self.models.append(model)
            
            elif self.learning_model == "HINNPerf":
                for i in range(estimators_num+1):
                    if i == estimators_num:
                        self.full_model,self.full_config = HINNPerf.HINNPerf(self.train_x,self.train_y,self.file_name)
                        break
                    np.random.seed(self.seed * i)
                    bagging_index = np.random.choice(lens,lens,replace=True)
                    if len(set(bagging_index)) == 1:
                        bagging_index = np.random.choice(lens,lens-1,replace=False)
                    bagging_x = [self.train_x[j] for j in bagging_index]
                    bagging_y = [self.train_y[j] for j in bagging_index]
                    model, config = HINNPerf.HINNPerf(bagging_x,bagging_y,self.file_name)
                    self.models.append(model)
                    self.best_configs.append(config)

    def predict(self, x,estimators_num = 10,return_std=True):
        if self.learning_model in ["RF","LR","KNN","SVR","KRR",'SPLConqueror',"DECART","DeepPerf"]:
            pred = []
            if self.funcname != "flash":
                for i in range(estimators_num):
                    try:
                        bagging_pred = self.models[i].predict(x)
                        # print(bagging_pred)
                    except ValueError:
                        bagging_pred = self.models[i].predict([[0.5]*len(x[0])])
                    pred.append(bagging_pred[0])
            try:
                result_pred = self.full_model.predict(x)
            except ValueError:
                result_pred = self.full_model.predict([[0.5]*len(x[0])])
            # print(result_pred,"---",x)
        else:
            if self.learning_model == "DaL":
                pred = []
                X = copy.deepcopy(x[0])
                X = np.divide(X, self.max_X)
                for j in range(self.N_features):  ## 预测在哪一个类
                    X[j] = X[j] * self.weights[j] 

                temp_cluster = self.forest.predict([X]) 
                if self.funcname != "flash":
                    # print(estimators_num)
                    for i in range(estimators_num):
                        for d in range(self.k): ## 匹配成功聚类之后进行预测
                            if d == temp_cluster:
                                if self.ks_DAL[d] < self.k or len(self.modelss[i])<self.k:
                                    pred.append(self.modelss[i][0].predict(np.divide(x, self.max_X))*self.max_Y)
                                else:
                                    pred.append(self.modelss[i][d].predict(np.divide(x, self.max_X))*self.max_Y)
                for d in range(self.k): ## 匹配成功聚类之后进行预测
                    if d == temp_cluster:
                            # result_pred = self.full_model[0].predict(x)*self.max_Y
                        result_pred = self.full_model[d].predict(np.divide(x, self.max_X))*self.max_Y
                        # print(result_pred)
                # print(result_pred,"---",x)

            if self.learning_model == "DaL_RF":
                pred = []
                X = copy.deepcopy(x[0])
                X = np.divide(X, self.max_X)
                for j in range(self.N_features):  ## 预测在哪一个类
                    X[j] = X[j] * self.weights[j] 
                # print(X)
                temp_cluster = self.forest.predict([X]) 
                if self.funcname != "flash":
                    for i in range(estimators_num):
                        for d in range(self.k): ## 匹配成功聚类之后进行预测
                            if d == temp_cluster:
                                if len(self.modelss[i]) == 1:

                                    pred.append(self.modelss[i][0].predict(np.divide(x, self.max_X))*self.max_Y)
                                else:

                                    pred.append(self.modelss[i][d].predict(np.divide(x, self.max_X))*self.max_Y)
                for d in range(self.k): ## 匹配成功聚类之后进行预测
                    if d == temp_cluster:
                            # result_pred = self.full_model[0].predict(x)*self.max_Y
                        result_pred = self.full_model[d].predict(np.divide(x, self.max_X))*self.max_Y

                # print(result_pred,"---",x)

            if self.learning_model == "HINNPerf":
                pred = []
                if self.funcname != "flash":
                    for i in range(estimators_num):
                        bagging_pred = self.models[i].predict(x,self.best_configs[i])
                        pred.append(bagging_pred[0])
            
                result_pred = self.full_model.predict(x,self.full_config) # runner对象不一样
                # print(result_pred)
                # print(result_pred,"---",x)
                  
        if self.learning_model == "GP":
            result_pred,result_std = self.full_model.predict(x, return_std=True)
            return result_pred, result_std
        if self.learning_model == "DT":
            estimators = self.full_model.estimators_
            pred = []
            for e in estimators:
                pred.append(e.predict(x))
            result_pred = self.full_model.predict(x)

            
        # print("pred: ",pred)
        pred = np.array(pred)
        if self.funcname != "flash":
            result_std = np.std(pred)
        else:
            result_std = 0
        if self.learning_model == "HINNPerf":
            result_pred = result_pred[0]
        if self.learning_model in ["HINNPerf","DaL","DeepPerf","Perf_AL"]:
            result_pred = result_pred[0]
        # print("error: ",result_std)
        # print(result_pred, result_std)
        return result_pred, result_std

    def save_model(self):
        if self.step%10 == 0 or self.funcname == "None":
            if self.learning_model in ["DT","RF","LR","DT","KNN","SVR","KRR","GP","SPLConqueror","DECART"]:
                f = open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step) + '.p', 'wb')
                pickle.dump(self.full_model, f)  # 修改
                f.close()
            elif self.learning_model in ["DaL_RF"]:
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_forest'+'.p',"wb")
                pickle.dump(self.forest,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_weight'+'.p',"wb")
                pickle.dump(self.weights,f)  
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_max_X'+'.p',"wb")
                pickle.dump(self.max_X,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_max_Y'+'.p',"wb")
                pickle.dump(self.max_Y,f) 
                f.close()      
                f = open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step) + '.p', 'wb')
                pickle.dump(self.full_model, f)  # 修改
                f.close()                
            elif self.learning_model in ["DaL"]:
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_configs'+'.p',"wb")
                pickle.dump(self.config,f)
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_dalx'+'.p',"wb")
                pickle.dump(self.dalx,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_daly'+'.p',"wb")
                pickle.dump(self.daly,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_lr_opt'+'.p',"wb")
                pickle.dump(self.lr_opt,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_forest'+'.p',"wb")
                pickle.dump(self.forest,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_weight'+'.p',"wb")
                pickle.dump(self.weights,f)  
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_max_X'+'.p',"wb")
                pickle.dump(self.max_X,f) 
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_max_Y'+'.p',"wb")
                pickle.dump(self.max_Y,f) 
                f.close()
            elif self.learning_model in["HINNPerf"]:
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_config'+'.p',"wb")
                pickle.dump(self.full_config,f)
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_indeps'+'.p',"wb")
                pickle.dump(self.train_x,f)
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_deps'+'.p',"wb")
                pickle.dump(self.train_y,f)
                f.close()
            elif self.learning_model in ["DeepPerf"]:
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_config'+'.p',"wb")
                pickle.dump(self.full_config,f)   
                f.close()   
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_lr_opt'+'.p',"wb")
                pickle.dump(self.lr_opt,f) 
                f.close() 
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_indeps'+'.p',"wb")
                pickle.dump(self.train_x,f)
                f.close()
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_deps'+'.p',"wb")
                pickle.dump(self.train_y,f)
                f.close()
            elif self.learning_model in ["Perf_AL"]:
                import torch
                torch.save(self.full_model,'./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step))
                f =open('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed) +'_'+'step'+str(self.step)+'_features_miny_maxy'+'.p',"wb")
                pickle.dump([self.AL_features,self.min_Y,self.max_Y],f) 
                f.close() 
            print('./Pickle_all/PickleLocker_'+str(self.funcname)+'_models'+ '/'+str(self.file_name)[:-4] +'/'+str(self.learning_model)+'_'+'seed'+str(self.seed)+'_'+'step'+str(self.step))
        