from batch import best_config,GGA,paramILS,GA,ConEx,rd,sampling,E_search
from sequential import boca,atconf,flash,ottertune,restune,robotune,smac,tuneful
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
    elif funcname == "E_search":
        xs, result, used_budget = E_search.run_irace(filename, model_name=modelname, seed = seed,maxlives=maxlives,budget=budget)
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

    x_axis = range(len(result)+1)[1:]  
    f = open('./Pickle_all/PickleLocker_'+str(funcname)+'_results'+ '/'+str(filename)[:-4] +'/'+str(modelname)+'_'+'seed'+str(seed)+'.csv','w',newline="")
    csv_writer = csv.writer(f)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    csv_writer.writerow(x_axis)
    csv_writer.writerow(xs)
    csv_writer.writerow(result)
    csv_writer.writerow([used_budget])
    f.close()

if __name__ == '__main__':
    import multiprocessing as mp
    seeds = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130]
    # modelname = "real"
    mp.freeze_support()
    pool = mp.Pool(processes=10)
    data = 'Data'
    modelnames = ["DT","RF","LR","SVR","GP","SPLConqueror","DECART","DeepPerf","HINNPerf","DaL_RF"]
    funcnames = ["boca","atconf","flash","ottertune","restune","robotune","smac","tuneful"]
    # add other system data (maximun need change requre func, > and <, and initial parameter)
    files = ['7z.csv'] 
    budgets = [382]
    for num in range(len(files)):
        budget = budgets[num]
        file = files[num]
        for seed in seeds:
            for model_name in modelnames:
                for funcname in funcnames:  
                    filename = data+'/'+file
                    if not os.path.exists('./Pickle_all/PickleLocker_'+str(funcname)+'_results'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed)+'.csv'):
                        print("-----------------------"+str(file)+"_"+str(seed)+"_"+str(funcname)+"_"+str(model_name)+"------------------------------") 
                        # pool.apply_async(run_main_free,(seed,data,file,model_name,funcname,budget)) 
                        run_main_free(seed,data,file,model_name,funcname,budget)
    pool.close()
    pool.join()
