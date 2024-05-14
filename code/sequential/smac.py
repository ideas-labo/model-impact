
from os import listdir
from random import shuffle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
import pickle
from scipy.stats import norm
from util import get_objective
from util import read_file
from util.bagging import bagging

def Result(configuration,model,eta,model_name):
    if model_name=="RF__":
        estimators = model.estimators_
        pred = []
        for e in estimators:
            pred.append(e.predict([configuration]))
        value = get_ei(pred, eta)
        return value
    else:
        value_pred,value_sigma = model.predict([configuration])
        value = ei(value_pred,value_sigma,eta)
        # print("VALUE: ",value)
        return value
    
def fixed_interval_sample(lst, n, keep_ends=True):
    interval = (len(lst) / (n-1))  
    indices = [0] + [int(i*interval) for i in range(1, n-1)]
    if keep_ends:
        indices.append(-1)
       
    sampled = [lst[i] for i in indices]
    
    return sampled

def find_neighbourhood(configuration,file):
    res = []
    for i in range(len(file.independent_set)):
        if len(file.independent_set[i])>10:
            tmp_independent_set = fixed_interval_sample(file.independent_set[i],10)
        else:
            tmp_independent_set = file.independent_set[i]
        for j in tmp_independent_set:
            tmp = configuration[:]
            tmp[i] = j
            res.append(tmp)
    shuffle(res)
    # print(len(res))
    return res

def iterative_first_improvement(configuration,model,eta,file,model_name):
    while 1:
        tmp_configuration = configuration
        for i in find_neighbourhood(tmp_configuration,file):
            if Result(i,model,eta,model_name) < Result(tmp_configuration,model,eta,model_name):
                configuration = i
                # print(get_objective_score(i))
                break
        if tmp_configuration == configuration:
            break
    return configuration

def model_randomforest(train_independent, train_dependent):

    # model = RandomForestRegressor(max_features=5/6,n_estimators=10,min_samples_split=10)
    model = RandomForestRegressor()
    model.fit(train_independent, train_dependent)

    return model

def get_ei(pred, eta):
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)

    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()

    return f

def ei(m,s,eta):
    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    if s==0:
        return 0
    else:
        f = calculate_f()
    return f

def get_training_sequence_by_smac(training_indep, training_dep, all_indep,eta,file,model_name,filename,seed,step):
    if model_name == "RF_skip":
        model = model_randomforest(training_indep, training_dep)
    else:
        model = bagging(train_x=training_indep,train_y=training_dep,learning_model=model_name,file_name=filename,seed=seed,step=step,funcname="smac")
        model.bagging_fit()
    eis = []
    for i in training_indep:
        value = Result(i,model,eta,model_name)
        ei_zip = [i, value]
        eis.append(ei_zip)
    # print(eis)
    ######## 替换模型修改这里 ############# 
    

    sort_merged_ei = sorted(eis, key=lambda x: x[1], reverse=True)[:10]
    top_10 = [i[0] for i in sort_merged_ei]

    inc_random10000 = random.sample(all_indep,1000 if len(all_indep)>1000 else len(all_indep))
    for i in top_10:
        inc_random10000.append(iterative_first_improvement(i,model,eta,file,model_name))
    

    return inc_random10000, model

def get_best_configuration_id_smac(training_indep, training_dep, all_indep, eta,file,model_name,filename,seed,step):
    test_sequence, model = get_training_sequence_by_smac(training_indep, training_dep, all_indep,eta,file,model_name,filename,seed,step)
    merged_ei = []
    if model_name == "RF11":

        estimators = model.estimators_
        for i in range(len(test_sequence)):
            pred = []
            independent = test_sequence[i]
            for e in estimators:
                pred.append(e.predict([independent]))
            value = get_ei(pred, eta)
            ei_zip = [independent, value]
            merged_ei.append(ei_zip)
        if step%10 == 0:
            f = open('./Pickle_all/PickleLocker_smac_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step) + '.p', 'wb')
            pickle.dump(model, f)  # 修改
            f.close()
    else:
        model.save_model()
        for i in range(len(test_sequence)):
            pred = []
            independent = test_sequence[i]
            value_pred,value_sigma = model.predict([independent])
            value = ei(value_pred,value_sigma,eta)
            ei_zip = [independent, value]
            merged_ei.append(ei_zip)
        
    sort_merged_ei = sorted(merged_ei, key=lambda x: x[1], reverse=True)
    x_max = sort_merged_ei[0][0]

    return(x_max)

def run_smac(filename, model_name="RF",initial_size=10, maxlives=100, budget=30,seed=0):
    steps = 0
    lives = maxlives
    file = read_file.get_data(filename, initial_size)
    training_dep = [t.objective[-1] for t in file.training_set]
    all_indep = [t.decision for t in file.all_set]
    result = 1e20
    for x in training_dep:
        if result > x:
            result = x
    results = []
    x_axis = []
    xs = []
    training_indep = [t.decision for t in file.training_set]
    training_dep = [t.objective[-1] for t in file.training_set]
    tuple_is = training_indep[:]
    best_loop = 0
    while initial_size + steps <= budget:
        steps += 1
        # print("len of training independent: ", len(training_indep))
        best_solution = get_best_configuration_id_smac(
            training_indep, training_dep, all_indep, result, file,model_name,filename,seed,steps)
        best_result,tuple_i = get_objective.get_objective_score_with_similarity(
            file.dict_search, best_solution)
        # print(best_result)
        training_indep.append(best_solution)
        training_dep.append(best_result)
        x_axis.append(steps)
        xs.append(tuple_i)
        results.append(best_result)
        if best_result < result:
            result = best_result
            lives = maxlives
            best_loop = steps
        else:
            lives -= 1
        if tuple_i in tuple_is:
            budget += 1
        else:
            tuple_is.append(tuple_i)
        print('loop: ', steps, 'reward: ', best_result)
        if lives == 0:
            break
    return np.array(xs), np.array(results), np.array(x_axis), result, best_loop, len(tuple_is)
