
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from util import read_file
from util import get_objective
from util.bagging import bagging
import pickle

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

def get_similar(dict_search, best_solution):
    min_dist = 1e20
    scaler = MinMaxScaler()
    scaler.fit_transform([list(i) for i in list(dict_search.keys())])
    for i in list(dict_search.keys()):
        a = scaler.transform([list(i)])
        b = scaler.transform([best_solution])
        tmp = distance(a,b)
        if tmp < min_dist:
            min_dist = tmp
            tmp_value = dict_search.get(tuple(i))
    return tmp_value
    
def round_num(num, discrete_list):
    max_num = 1e20
    return_num = 0
    num = (max(discrete_list)-min(discrete_list))*num+min(discrete_list)
    for i in discrete_list:
        if abs(num-i) < max_num:
            max_num = abs(num-i)
            return_num = i
    return return_num



def run_restune(initial_size, filename, model_name="GP",acqf_name = 'EI',budget=20,seed=0,maxlives=100):
    lives = maxlives
    results = []
    x_axis = []
    xs = []
    memory = ReplayMemory()
    num_collections = initial_size  # 多少个已知量
    file = read_file.get_data(filename, initial_size)
    header , features , target = read_file.load_features(file)
    best_result = 1e20
    x_all = [t.decision for t in file.all_set]
    global_step = 0
    tuple_is = [t.decision for t in file.training_set]
    if acqf_name == "EI":
        def ac(x, model, y_min, xi=0):
            mean, std = model.predict(x, return_std=True)
            # print(std)
            z = (-mean +y_min - xi)/std
            return (-mean + y_min - xi) * norm.cdf(z) + std * norm.pdf(z)

    for i in range(num_collections):
        action = file.training_set[i]
        # reward, _ = env.simulate(action)
        reward = action.objective[-1]
        # print(reward)
        memory.push(np.array(action.decision), np.array([reward]))
   
    # 主循环  
    lens =len(header)

    pbounds = {}
    for name in range(lens):
        pbounds[name] = (0, 1)
    bounds = np.array(list(pbounds.values()), dtype=np.float64)
    Scaler_x = MinMaxScaler().fit(x_all)
    # features = scaler.fit_transform(features)

    while global_step+initial_size <= budget:
        global_step += 1
        lives -= 1
        actions, rewards = memory.get_all()
        X_scaled = Scaler_x.transform(actions)
        train_x = X_scaled.astype(np.float64)
        Scaler_y = StandardScaler().fit(rewards)
        train_y = Scaler_y.transform(rewards)
        
        if global_step%10 == 0:
            f = open('./Pickle_all/PickleLocker_restune_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step)+'_scaler' + '.p', 'wb')
            pickle.dump((Scaler_x,Scaler_y), f)
            f.close()



        if model_name == "GP":
            model = GaussianProcessRegressor()
            model.fit(train_x,train_y)
            
            if global_step%10 == 0:
                f = open('./Pickle_all/PickleLocker_restune_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step) + '.p', 'wb')
                pickle.dump(model, f)  # 修改
                f.close()
        else:
            model = bagging(train_x=train_x.tolist(),train_y=train_y.tolist(),learning_model=model_name,file_name=filename,seed=seed,step=global_step,funcname="restune")
            model.bagging_fit()
            model.save_model()
        
        y_min = min(train_y)

        max_acq = None

        # x_seeds = random.sample(features,10)  ## 模仿optimize_acqf的默认参数，少了十次重启
        x_seeds = np.random.RandomState(global_step).uniform(bounds[:, 0], bounds[:, 1],
                                 size=(20, bounds.shape[0]))
        tmps = []
        for x_try in x_seeds:
            # print(x_try)
            
            res = minimize(lambda x: -ac(x.reshape(1, -1), model=model, y_min=y_min),
                    np.array(x_try),
                    bounds=bounds,
                    method="L-BFGS-B")
            
            if not res.success:
                continue
            # res_list.append(res)
            tmp = (res.x,res.fun)
            tmps.append(tmp)
            tmps.append((x_try,-ac(x_try.reshape(1,-1),model=model,y_min=y_min)))
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun
                # print(res.fun)
        # print("max_acq: ",max_acq)
        # sort_merged_ei = sorted(tmps, key=lambda x: x[1], reverse=False)
        # print(sort_merged_ei)
        # for i,j in sort_merged_ei:
        #     scaler_i = Scaler_x.inverse_transform([i])
        #     origin_i = get_similar_x(file.dict_search,scaler_i[0])
        #     if origin_i not in actions and origin_i not in xs:
        #         x_max = i
        #         break

        x_max = np.clip(x_max, bounds[:, 0], bounds[:, 1])
        x_max = list(x_max)
       
        for i in range(len(x_max)):
            x_max[i] = (max(file.independent_set[i])-min(file.independent_set[i]))*x_max[i] + min(file.independent_set[i])

        reward,tuple_i = get_objective.get_objective_score_with_similarity(file.dict_search, x_max)
        # x_max = get_similar_x(file.dict_search,x_max)
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

    return np.array(xs), np.array(results), np.array(x_axis), best_result, best_loop, len(tuple_is)