
from model_free import best_config,GGA,irace,paramILS,GA,ConEx,rd,sampling
from moduces import boca,atconf,flash,ottertune,restune,robotune,smac,tuneful
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import csv
from util import bagging
import os

def train_model(train_x,train_y,modelname,filename,seed):
    model = bagging(train_x,train_y,learning_model=modelname,file_name=filename,seed=seed,step="NULL",funcname="NULL")
    model.bagging_fit(estimators_num=0)
    return model


def run_main_free(seed,data,file,modelname,funcname,budget):
    

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    filename = data+'/'+file
    print(filename)

    maxlives = 200

    if funcname == "ConEx":
        xs, result, used_budget = ConEx.run_conex(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "sampling":
        xs, result, used_budget = sampling.run_sampling(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "random":
        xs, result, used_budget = rd.run_random(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "GA":
        xs, result, used_budget = GA.run_ga(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "best_config":
        xs, result, used_budget = best_config.run_best_config(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "GGA":
        xs, result, used_budget = GGA.run_gga(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "irace":
        xs, result, used_budget = irace.run_irace(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "paramILS":
        xs, result, used_budget = paramILS.run_paramILS(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
    elif funcname == "boca":
        xs, result, x_axis, _, _, used_budget = boca.run_boca(initial_size=20, filename=filename, maxlives=maxlives, budget=budget, fnum=2, rnum0=2**6,seed=seed,model_name=modelname)
    elif funcname == "atconf":
        xs, result, x_axis, _, _, used_budget = atconf.run_atconf(initial_size=20,filename=filename,  acqf_name="LCB",budget=budget, features_num=9,seed=seed, maxlives=maxlives,model_name=modelname)
    elif funcname == "flash":
        xs, result, x_axis, _, _, used_budget = flash.run_flash(initial_size=20,filename=filename,seed=seed,budget=budget, maxlives=maxlives,model_name=modelname)
    elif funcname == "ottertune":
        xs, result, x_axis, _, _, used_budget = ottertune.run_ottertune(initial_size=20, filename=filename, acqf_name = "UCB",budget=budget, Beta=0.3,features_num=6,seed=seed, maxlives=maxlives,model_name=modelname)
    elif funcname == "restune":
        xs, result, x_axis, _, _, used_budget = restune.run_restune(initial_size=20, filename=filename,  acqf_name = "EI", budget=budget,seed=seed, maxlives=maxlives,model_name=modelname)
    elif funcname == "robotune":
        xs, result, x_axis, _, _, used_budget = robotune.run_robotune(initial_size=20, filename=filename,  budget=budget,seed=seed, maxlives=maxlives,model_name=modelname)
    elif funcname == "smac":
        xs, result, x_axis, _, _, used_budget = smac.run_smac(initial_size=20, filename=filename, maxlives=maxlives,budget=budget,seed=seed,model_name=modelname)
    elif funcname == "tuneful":
        xs, result, x_axis, _, _, used_budget = tuneful.run_tuneful(initial_size=20, filename=filename, fraction_ratio=0.9,budget=budget,seed=seed, maxlives=maxlives,model_name=modelname)
    else:
        return "ERROR!!"

    x_axis = range(len(result)+1)[1:]  # 创建索引
    # 1. 创建文件对象
    f = open('./Pickle_all/PickleLocker_'+str(funcname)+'_results'+ '/'+str(filename)[:-4] +'/'+str(modelname)+'_'+'seed'+str(seed)+'.csv','w',newline="")
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    csv_writer.writerow(x_axis)
    csv_writer.writerow(xs)
    csv_writer.writerow(result)
    csv_writer.writerow([used_budget])
    # 5. 关闭文件
    f.close()

if __name__ == '__main__':
    import multiprocessing as mp
    seeds = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130]
    # modelname = "real"
    mp.freeze_support()
    pool = mp.Pool(processes=10)
    modelnames = ["HINNPerf"]
    funcnames = ["boca","atconf","flash","ottertune","restune","robotune","smac","tuneful"]
    files = ['7z.csv','Apache.csv','dconvert.csv','deeparch.csv','exastencils.csv','Hadoop.csv','hipacc.csv','hsmgp.csv','javagc.csv','jump3r.csv',
    'kanzi.csv','MariaDB.csv','MongoDB.csv','polly.csv','PostgreSQL.csv','redis.csv','SaC.csv','spark.csv','SQL.csv','storm.csv',
    'tomcat.csv','vp9.csv','xgboost.csv','BDBC_AllNumeric.csv','brotli.csv','HSQLDB.csv','LLVM_AllNumeric.csv','lrzip.csv','noc-CM-log.csv',
            'sort-256.csv','wc-c4-3d.csv']
    # files = ["Apache.csv"]
            ## 去掉了一些数据集 xz，batik，trimesh,x264 加 去掉apache_allnumeric,sqlite    
    budgets = [382, 271, 335, 207, 416, 297, 371, 218, 289, 232,
     237, 226, 278, 285, 298, 298, 316, 326, 206, 263, 
     282, 271, 278, 259, 203, 149, 182, 184, 129, 
     127, 230]
    for num in range(len(files)):
        
        budget = budgets[num]
        file = files[num]
        if file in ['7z.csv','Apache.csv','batik.csv','dconvert.csv','deeparch.csv','exastencils.csv','Hadoop.csv','hipacc.csv',
                    'hsmgp.csv','javagc.csv','jump3r.csv','kanzi.csv','MariaDB.csv','MongoDB.csv','polly.csv','PostgreSQL.csv','redis.csv','SaC.csv',
                    'spark.csv','SQL.csv','storm.csv','tomcat.csv','Trimesh.csv','vp9.csv','x264.csv','xgboost.csv','xz.csv']:
            data = 'Data_big'
        else:
            data = 'Data_small'

        for seed in seeds:
            for model_name in modelnames:
                for funcname in funcnames:  
                    
                    filename = data+'/'+file
                    if not os.path.exists('./new_pickle/101-110hinn/Pickle_all/PickleLocker_'+str(funcname)+'_results'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed)+'.csv'):
                        print("-----------------------"+str(file)+"_"+str(seed)+"_"+str(funcname)+"_"+str(model_name)+"------------------------------") 
                        # pool.apply_async(run_main_free,(seed,data,file,model_name,funcname,budget)) 
                        # run_main_free(seed,data,file,model_name,funcname,budget)
    pool.close()
    pool.join()



# import time
# start = time.time()
# run_main_free(106,"Data_big","jump3r.csv","LR","robotune",232)
# end = time.time()
# print(end-start)
# seed = 101
# file = 'Apache_AllNumeric.csv'
# budget = 133
# data = 'Data_small'
# model_name = 'DaL'
# funcname = 'boca'
# print("-----------------------"+str(file)+"_"+str(seed)+"_"+str(funcname)+"_"+str(model_name)+"------------------------------")
# run_main_free(seed,data,file,model_name,funcname,budget)



# if __name__ == '__main__':
#     import multiprocessing as mp
#     file = 'Apache_AllNumeric.csv'   
#     data = 'Data_small'
#     seed = 34
#     modelname = "real"
#     # mp.freeze_support()
#     # pool = mp.Pool()
#     for funcname in ["irace"]:#["boca","atconf","flash","ottertune","restune","robotune","smac","tuneful",'BOHB','HB','best_config','GGA','irace','paramILS']: 
#         run_main_free(seed,data,file,modelname,funcname)
 

# if __name__ == '__main__':
#     import multiprocessing as mp
#     for file in ['Apache_AllNumeric.csv','BDBC_AllNumeric.csv','brotli.csv','HSQLDB.csv','LLVM_AllNumeric.csv','lrzip.csv','noc-CM-log.csv','sort-256.csv','sqlite.csv','wc-c4-3d.csv']:
#         print("-----------------------"+str(file)+"------------------------------")
#         data = 'Data_small'
#         seeds = [31,32,33,34,35]
#         modelname = "real"
#         mp.freeze_support()
#         pool = mp.Pool(processes=50)
#         for seed in seeds:
#             for funcname in ['BOHB','HB']:  
#                 pool.apply_async(run_main_free,(seed,data,file,modelname,funcname)) 
#         pool.close()
#         pool.join()
