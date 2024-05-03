
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import random
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from util import get_objective
from util import read_file
from util.bagging import bagging
import pickle

def choose(feature_count,feature_sort,now_best,unimportant_features,dropout_mix_p=0.8):
    feature_dict={}
    x=random.random()
    if x < dropout_mix_p:
        dropout_fill = 'copy'
    else:
        dropout_fill = 'rand'
    print('show dropout_fill')
    print(dropout_fill)

    if dropout_fill == 'rand':
        for f in unimportant_features:
            tmp = feature_count[feature_sort[f]]
            x = random.sample(tmp,1)
            feature_dict[f] = float(x[0])
        return feature_dict
    else:
        for f in unimportant_features:
            x = now_best[feature_sort[f]]
            feature_dict[f] = x
        return feature_dict
            

# 清理不满足的数据
def clean_data(important_features, file, now_best):
    unimportant_features = []
    for i in file.features:
        if i not in important_features:
            unimportant_features.append(i)
    feature_sort = {x: i for i, x in enumerate(file.features)}  # 特征:特征编号
    # 转置
    features = [t.decision for t in file.all_set]
    feature_count = list(map(list, zip(*features)))
    # 得到出现最多的数
    feature_dict = choose(feature_count,feature_sort,now_best,unimportant_features)
    # print(feature_dict.keys())
    # print(feature_dict.values())

    # 把file里面的元素删了
    # print(unimportant_features)
    for i in [file.all_set, file.training_set, file.testing_set]:
        for feature in unimportant_features:
            for j in i:
                # print(feature_dict)
                # print(feature_sort)
                if j.decision[feature_sort[feature]] != feature_dict[feature]:
                    i.remove(j)

    # 修改去重的自变量
    for feature in unimportant_features:
        file.independent_set[feature_sort[feature]] = [feature_dict[feature]]
    # print(file.independent_set)
    return unimportant_features


def round_num(num, discrete_list):
    max_num = 1e20
    return_num = 0
    for i in discrete_list:
        if abs(num - i) < max_num:
            max_num = abs(num - i)
            return_num = i
    return return_num


def acq_max(ac, model, y_max, bounds, random_state, n_warmup=1000, n_iter=20):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    if ac == "LCB":
        def ac(x,y_max,model,kappa=2.576):
            # mean, std = model.predict(x, return_std=True)
            mean, std = model.predict(x,return_std = True)
            return mean - kappa * std
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_warmup, bounds.shape[0]))
    ys = []
    for i in x_tries:
        ys.append(ac(x=[i], model=model, y_max=y_max))
    x_min = x_tries[ys.index(min(ys))]
    min_acq = min(ys)
    # print(max_acq)
    # print(max_acq)
    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_iter, bounds.shape[0]))
    tmps = []
    for x_try in x_seeds:
        # print(x_try)
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: ac(x.reshape(1, -1), model=model, y_max=y_max),
                        np.array(x_try),
                        bounds=bounds,
                        method="L-BFGS-B")
        # print(res.fun)
        # See if success
        if not res.success:
            continue

        # print(res.fun)
        # Store it if better than previous minimum(maximum).
        if min_acq is None or res.fun <= min_acq:
            x_min = res.x
            min_acq = res.fun

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.

    return np.clip(x_min, bounds[:, 0], bounds[:, 1])

def recover(x_new, header,now_best,unimportant_features,features):
    feature_sort = {x: i for i, x in enumerate(header)}  # 特征:特征编号

    # 得到出现最多的数
    feature_dict = choose(features,feature_sort,now_best,unimportant_features)
    for feature in feature_dict:
        x_new.insert(feature_sort[feature],feature_dict[feature])
    return x_new
        


def run_atconf(filename,initial_size=10, model_name="LR", acqf_name = "LCB",budget=20, features_num=9,seed=0, maxlives=100):

    file = read_file.get_data(filename, initial_size)
    x_init = [t.decision for t in file.training_set]
    y_init = [[t.objective[-1]] for t in file.training_set]
    x_all = [t.decision for t in file.all_set]
    y_min = 1e20
    for i in y_init:
        if i[0] < y_min:
            y_min = i[0]
    pbounds = {}
    step = 0
    # initial_num = 5
    header, features, y = read_file.load_features(file)
    best_loop = -1
    features_num = int(9/10*len(header))
    for name in range(features_num):
        pbounds[name] = (0, 1)
    bounds = np.array(list(pbounds.values()), dtype=np.float64)
    results = []
    x_axis = []
    xs = []
    tuple_is = x_init[:]
    Scaler_x = MinMaxScaler().fit(x_all)
    lives = maxlives

    while step+initial_size <= budget:
        step += 1
        lives -= 1
        if lives == -1:
            break
        d_features = random.sample(header, features_num)
        unimportant_features_id = []
        min_index = np.argmin(y_init)
        now_best = x_init[min_index]
        unimportant_features = []

        for i in header:
            if i not in d_features:
                unimportant_features.append(i)
        for i, j in enumerate(header):
            if j in unimportant_features:
                unimportant_features_id.append(i)
        
        # 删除未选择特征
        print(unimportant_features_id)

        
        x_train = Scaler_x.transform(x_init)
        Scaler_y = StandardScaler().fit(y_init)
        y_train = Scaler_y.transform(y_init)
        # print(y_train)
        # print(Scaler_y.inverse_transform(y_train))
        
        if step%10 == 0:
            f = open('./Pickle_all/PickleLocker_atconf_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step)+'_scaler' + '.p', 'wb')
            pickle.dump((Scaler_x,Scaler_y,unimportant_features_id), f)
            f.close()



        x_train = np.delete(x_train, unimportant_features_id, axis=1)

        if model_name == "GP":
            model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=2
            )
            model.fit(x_train, y_train)
            
            if step%10 == 0:
                f = open('./Pickle_all/PickleLocker_atconf_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step) + '.p', 'wb')
                pickle.dump(model, f)  # 修改
                f.close()
        else:
            model = bagging(x_train,y_train,learning_model=model_name,file_name=filename,seed=seed,step=step,funcname="atconf")
            model.bagging_fit()
            model.save_model()

        y_max = max(y_train)
        # print(bounds)
        x_new = acq_max(ac=acqf_name, model=model, y_max=y_max, bounds=bounds, random_state=np.random.RandomState(step))
        
        x_new = x_new.tolist()
    
        x_new = recover(x_new, header, now_best, unimportant_features, file.independent_set)
        for i in range(len(x_new)):
            x_new[i] = (max(file.independent_set[i])-min(file.independent_set[i]))*x_new[i] + min(file.independent_set[i])
     
        y_new,tuple_i = get_objective.get_objective_score_with_similarity(file.dict_search, x_new)
        
        if tuple_i in tuple_is:
            budget += 1
        else:
            tuple_is.append(tuple_i)
        # print(tuple_is)
        x_init.append(x_new)
        y_init.append([y_new])
        results.append(y_new)
        if y_new < y_min:
            y_min = y_new
            best_loop = step
            lives = maxlives
        x_axis.append(step)
        xs.append(tuple_i)
        print('loop: ', step, 'now best: ',y_min, 'reward: ', y_new)

    return np.array(xs), np.array(results), np.array(x_axis), y_min, best_loop, len(tuple_is)
