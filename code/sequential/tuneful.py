
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import time
from util import get_objective
from util import read_file
from util.bagging import bagging
from scipy.optimize import minimize
from scipy.stats import norm


class ReplayMemory(object):

    def __init__(self):
        self.actions = []
        self.rewards = []

    def push(self, action, reward):
        self.actions.append(action.tolist())
        self.rewards.append(reward.tolist())

    def get_all(self):
        return self.actions, self.rewards

def distance(p1,p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	dist_orig = np.sqrt(np.sum(np.square(p1-p2)))

	return dist_orig

def get_similar_x(dict_search, best_solution):
    min_dist = 1e20
    scaler = MinMaxScaler()
    scaler.fit_transform([list(i) for i in list(dict_search.keys())])
    for i in list(dict_search.keys()):
        a = scaler.transform([list(i)])
        b = scaler.transform([best_solution])
        tmp = distance(a,b)
        if tmp < min_dist:
            min_dist = tmp
            tmp_value = i
    return tmp_value

def get_ei(m, s, eta):

    s = sqrt(s)
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

def scale_data(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler ()
    # print (">>>>data to scale >>>>" , data)
    scaled_df =  scaler.fit_transform (data )

    #  print ("scaled DF >> " , scaled_df [:,11])
    return scaled_df

def get_sig_params (header , features , target , fraction):
    start_time = int(time.time())
    search_time = 0
    from sklearn.feature_selection import VarianceThreshold
    scaled_features = scale_data (features) 
    #print ("length of features  >>>> "  , features.shape)  
    # fit an Extra Trees model to the data
    sig_conf_all_iterations = [] 
    n_iterations = 30
    model = ExtraTreesRegressor ()
    error_all_itr=[]

    n_params  = int(float(fraction) *len (header))
    n_all_params = len (header)
    ####### build the model 100 time to overcome model randomness #########
    for x in range (n_iterations):
        all_params = [0] * n_all_params
        model.fit (scaled_features , target)
        normalized_importance =  100 * (model.feature_importances_ /max (model.feature_importances_)) 
        indices  = normalized_importance.argsort()[-n_params:][::-1]
        indices = np.array (indices)
        all_params = np.array (all_params)
        all_params [indices] = 1  # set the indices of the selected params to 1
    #        print ("all_params >>>> " , all_params)
        sig_conf_all_iterations= np.append(sig_conf_all_iterations , all_params)
    sig_conf_all_iterations = np.reshape (sig_conf_all_iterations ,  (n_iterations , n_all_params))
    sig_conf_all_iterations = np.count_nonzero (sig_conf_all_iterations , axis = 0)    # count the occurances of each param in the sig params over all the interations
    header = np.array (header) 
    indices = sig_conf_all_iterations.argsort()[-n_params:][::-1] # select the params that have the most occurances in the sig params over all the interations
    indices = np.array (indices)
    # print (">>>>>>>>>>>>" , indices)
    h_inf_params = header[indices] 
    # print (">>>>>>>>>" , h_inf_params)
    search_time += (int(time.time())- start_time)
    return  h_inf_params.tolist() 

def clean_data(important_features,file):
    unimportant_features = []
    for i in file.features:
        if i not in important_features:
            unimportant_features.append(i)
    feature_sort = {x:i for i, x in enumerate(file.features)}  
    features = [t.decision for t in file.all_set]
    feature_count = list(map(list, zip(*features)))
    feature_dict = {}
    for f in unimportant_features:
        tmp = feature_count[feature_sort[f]]
        x = max(set(tmp),key=tmp.count)
        feature_dict[f] = x
    # print(feature_dict.keys())
    # print(feature_dict.values())

    for i in [file.all_set,file.training_set,file.testing_set]:
        for feature in unimportant_features:
            for j in i:
                if j.decision[feature_sort[feature]] != feature_dict[feature]:
                    i.remove(j)
    
    for feature in unimportant_features:
        file.independent_set[feature_sort[feature]] = [feature_dict[feature]]
    # print(file.independent_set)

    return file
    
def reset_data(file,initial_size):
    indexes = range(len(file.all_set))
    train_indexes, test_indexes = indexes[:initial_size],  indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes)== len(indexes)), "Something is wrong"
    file.training_set = [file.all_set[i] for i in train_indexes]
    file.testing_set = [file.all_set[i] for i in test_indexes]
    return file

def ac(x, model, y_min, xi=0):
    mean, std = model.predict(x, return_std=True)
    z = (-mean +y_min - xi)/std
    return (-mean + y_min - xi) * norm.cdf(z) + std * norm.pdf(z)

def run_gpr_ei(initial_size, file, filename, budget=10,model_name="GP",seed=0,maxlives=100):
    lives = maxlives
    results = []
    x_axis = []
    xs = []
    memory = ReplayMemory()
    num_collections = initial_size  
    header,_ ,_= read_file.load_features(file)
    lens = len(header)
    best_result = 1e20

    tuple_is = []
    for i in range(num_collections):
        action = file.training_set[i]
        # reward, _ = env.simulate(action)
        reward = action.objective[-1]
        memory.push(np.array(action.decision), np.array([reward]))
        tuple_is.append(list(action.decision))
    pbounds = {}
    for name in range(lens):
        pbounds[name] = (0, 1)
    bounds = np.array(list(pbounds.values()), dtype=np.float64)
    # best_ei = 0
    global_step=0
    test_sequence = file.testing_set
    test_sequence = [i.decision for i in test_sequence]
    x_all = [t.decision for t in file.all_set]
    Scaler_x = MinMaxScaler().fit(x_all)


    while global_step+initial_size < budget:
        lives -= 1
        global_step += 1
        actions, rewards = memory.get_all()

        train_x = Scaler_x.transform(actions)
        Scaler_y = StandardScaler().fit(rewards)
        train_y = Scaler_y.transform(rewards)
        
        if global_step%10 == 0:
            f = open('./Pickle_all/PickleLocker_tuneful_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step)+'_scaler' + '.p', 'wb')
            pickle.dump((Scaler_x,Scaler_y), f)
            f.close()

        if model_name == "GP":
            model = GaussianProcessRegressor()
            model.fit(train_x, train_y)
            
            if global_step%10 == 0:
                f = open('./Pickle_all/PickleLocker_tuneful_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step) + '.p', 'wb')
                pickle.dump(model, f)  # 修改
                f.close()
        else:
            model = bagging(train_x=train_x.tolist(),train_y=train_y.tolist(),learning_model=model_name,file_name=filename,seed=seed,step=global_step,funcname="tuneful")
            model.bagging_fit()
            model.save_model()
        best_f = min(train_y)
        # print("best_f: ",best_f)
        max_acq = None
        i = 0
        x_seeds = np.random.RandomState(global_step).uniform(bounds[:, 0], bounds[:, 1],
                                 size=(20, bounds.shape[0]))
        # print(x_seeds)
        tmps = []
        for x_try in x_seeds:
            # print(x_try)
            
            res = minimize(lambda x: -ac(x.reshape(1, -1), model=model, y_min=best_f),
                    np.array(x_try),
                    bounds=bounds,
                    method="L-BFGS-B")
            
            if not res.success:
                continue
            # res_list.append(res)
            
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun

        x_max = list(x_max)
        for i in range(len(x_max)):
            x_max[i] = (max(file.independent_set[i])-min(file.independent_set[i]))*x_max[i] + min(file.independent_set[i])
        # print(x_max)
        reward,tuple_i = get_objective.get_objective_score_with_similarity(file.dict_search, x_max)
        memory.push(np.array(x_max), np.array([reward]))

        if reward < best_result:
            best_result = reward
            best_loop = global_step
            lives = maxlives

        if tuple_i in tuple_is:
            budget += 1
        else:
            tuple_is.append(tuple_i)
        results.append(reward)
        x_axis.append(global_step)
        xs.append(tuple_i)
        print('loop: ', global_step, 'reward: ', reward)
        if lives == 0:
            break

    return np.array(xs),np.array(results), np.array(x_axis), best_result,best_loop,len(tuple_is)

def run_tuneful(initial_size ,filename , model_name="GP",fraction_ratio=0.9,budget=20,seed=0,maxlives=100):
    # filename = "./Data/Dune.csv"
    file_data = read_file.get_data(filename, initial_size)
    header, features, target = read_file.load_features(file_data)
    features = features[:100]
    target = target[0:100]
    important_features = get_sig_params(header, features, target, fraction=fraction_ratio)
    file_data = clean_data(important_features, file_data)
    file_data = reset_data(file_data, initial_size)
    xs,results, x_axis, best_result,best_loop, used_budget = run_gpr_ei(initial_size, file_data,filename, budget,model_name,seed=seed,maxlives=maxlives)
    return np.array(xs), np.array(results), np.array(x_axis), best_result, best_loop, used_budget

