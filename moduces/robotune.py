from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import minimize
from util import read_file
from util import get_objective
from util.bagging import bagging

from collections import defaultdict
from doepy import build
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

def round_num(num, discrete_list):
    max_num = 1e20
    return_num = 0
    num = (max(discrete_list)-min(discrete_list))*num+min(discrete_list)
    for i in discrete_list:
        if abs(num-i) < max_num:
            max_num = abs(num-i)
            return_num = i
    return return_num

def clean_data(important_features,file):
    unimportant_features = []
    for i in file.features:
        if i not in important_features:
            unimportant_features.append(i)
    feature_sort = {x:i for i, x in enumerate(file.features)}  # 特征:特征编号
    # 转置
    features = [t.decision for t in file.all_set]
    feature_count = list(map(list, zip(*features)))
    # 得到出现最多的数
    feature_dict = {}
    for f in unimportant_features:
        tmp = feature_count[feature_sort[f]]
        x = max(set(tmp),key=tmp.count)
        feature_dict[f] = x
    print(feature_dict.keys())
    print(feature_dict.values())

    # 把file里面的元素删了
    for i in [file.all_set,file.training_set,file.testing_set]:
        for feature in unimportant_features:
            for j in i:
                if j.decision[feature_sort[feature]] != feature_dict[feature]:
                    i.remove(j)
    
    # 修改去重的自变量
    for feature in unimportant_features:
        file.independent_set[feature_sort[feature]] = [feature_dict[feature]]
    # print(file.independent_set)
    return file

def get_sig_params(header, features, target, num):


    X = np.array(features)
    y = target
    names = header

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建随机森林模型并拟合
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf = RandomForestRegressor()
    X_train = X_train[:100]
    y_train = y_train[:100]
    rf.fit(X_train, y_train)

    # 计算原始预测准确性
    y_pred = rf.predict(X_test)
    acc = r2_score(y_test, y_pred)

    # 初始化 MDA 字典
    scores = defaultdict(list)

    # 对每个特征进行打乱并计算 MDA
    for i in range(10):
        for i in range(len(X[0])):
            X_t = X_test.copy()
            random.shuffle(X_t[i])
            y_t = rf.predict(X_t)
            shuff_acc = r2_score(y_test, y_t)
            scores[names[i]].append((acc - shuff_acc) / acc)

    # 输出结果
    print("Features sorted by their score:")
    print(sorted([[round(np.mean(score), 4), feat] for feat, score in scores.items()], reverse=True))
    return  [i[1] for i in sorted([[round(np.mean(score), 4), feat] for feat, score in scores.items()], reverse=True)][:num+1]         

def Latin_sample(file, num_samples = 20):
    # filename = "./Data/Dune.csv"
    # file = get_data(filename)
    # header, features, target = load_features(file)

    def round_num(num, discrete_list):
        max_num = 1e20
        return_num = 0
        for i in discrete_list:
            if abs(num-i) < max_num:
                max_num = abs(num-i)
                return_num = i
        return return_num

    variables = {}
    bounds = file.independent_set
   
    for i in range(len(bounds)):
        if len(bounds[i]) == 1:
            # print(bounds[i])
            bounds[i].append(1.1) # 加一个数凑出范围，不影响结果，会近似

    for i in range(len(file.independent_set)):
        variables[file.features[i]] = bounds[i]
    sample = build.space_filling_lhs(variables, num_samples)

    data_array = np.array(sample)
    # 然后转化为list形式

    data_list = data_array.tolist()

    copy = data_list.copy()

    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            data_list[i][j] = round_num(data_list[i][j],file.independent_set[j])

    return data_list

def  gp_hedge(gains,y_opt,model,file,global_step,Scaler_x,actions,xs):
    
    def random_pick(some_list,probabilities):
        x = random.uniform(0,1)
        cumulative_probability=0.0
        for item,item_probability in zip(some_list,probabilities):
            cumulative_probability+=item_probability
            if x < cumulative_probability:
                break
        return item


    header , features , target = read_file.load_features(file)
    lens =len(header)
    pbounds = {}
    for name in range(lens):
        pbounds[name] = (0, 1)
    bounds = np.array(list(pbounds.values()), dtype=np.float64)
    x_maxs = []
    names = ["LCB","EI","PI"]
    tmpss= []
    for name in names:
        max_acq = None
        tmps= []
        
        if name == "LCB":
            def ac(model,X,kappa=0):
                mu, std = model.predict(X, return_std=True)
                return -mu + kappa * std
        elif name == "EI":
            def ac(model,X,y_opt):
                mu, std = model.predict(X, return_std=True)
                improve = y_opt - mu
                scaled = improve / std
                cdf = norm.cdf(scaled)
                pdf = norm.pdf(scaled)
                exploit = improve * cdf
                explore = std * pdf
                return exploit + explore
        else:
            def ac(model,X,y_opt):
                mu, std = model.predict(X, return_std=True)
                improve = y_opt - mu
                scaled = improve / std
                return norm.cdf(scaled)
      
        x_seeds = np.random.RandomState(global_step).uniform(bounds[:, 0], bounds[:, 1],
                                 size=(20, bounds.shape[0]))
        for x_try in x_seeds: 

            res = minimize(lambda x: -ac(model, x.reshape(1, -1),y_opt),
                            np.array(x_try),
                            bounds=bounds,
                            method="L-BFGS-B")

            if not res.success:
                continue

            if max_acq is None or res.fun <= max_acq:
                x_max = res.x
                max_acq = res.fun
                # print(res.fun)
        
            # tmp = (res.x,res.fun)
            # tmps.append(tmp)
            # tmp = (x_try,-ac(model, x_try.reshape(1, -1),y_opt))
            # tmps.append(tmp)
        # sort_merged_ei = sorted(tmps, key=lambda x: x[1], reverse=False)
        # print(sort_merged_ei)
        # for i,j in sort_merged_ei:
        #     scaler_i = Scaler_x.inverse_transform([i])
        #     origin_i = get_similar_x(file.dict_search,scaler_i[0])
        #     # print(xs)
        #     if origin_i not in actions and origin_i not in xs:
        #         x_max = i
        #         break

        x_maxs.append(x_max)

    logits = np.array(gains)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    next_x = random_pick(x_maxs,list(probs))
    return next_x,x_maxs

def run_robotune(initial_size, filename, model_name="GP",budget=20,seed=0,maxlives= 100):
    lives = maxlives
    results = []
    x_axis = []
    xs = []
    
    global_step = 0
    memory = ReplayMemory()
    num_collections = initial_size  # 多少个已知量
    file = read_file.get_data(filename)
    header, features, target = read_file.load_features(file)
    num = int(9/10*len(header))
    important_features = get_sig_params(header, features, target, num = num)
    file = clean_data(important_features, file)
    tuple_is = []
    best_result = 1e20
    actions = Latin_sample(file,num_collections)
    for action in actions:
        reward,tuple_i = get_objective.get_objective_score_with_similarity(file.dict_search,action)
        memory.push(np.array(action),np.array([reward]))
        tuple_is.append(tuple_i)

    # 主循环  
    lens =len(header)
 
    x_all = [t.decision for t in file.all_set]
    Scaler_x = MinMaxScaler().fit(x_all)
    gains = np.zeros(3)
    while global_step+initial_size <= budget:
        global_step += 1
        lives -= 1
        actions, rewards = memory.get_all()
        # print(actions)
        X_scaled = Scaler_x.transform(actions)
        train_x = X_scaled.astype(np.float64)
        Scaler_y = StandardScaler().fit(rewards)
        train_y = Scaler_y.transform(rewards)
        
        if global_step%10 == 0:
            f = open('./Pickle_all/PickleLocker_robotune_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step)+'_scaler' + '.p', 'wb')
            pickle.dump((Scaler_x,Scaler_y), f)
            f.close()

        if model_name == "GP":
            model = GaussianProcessRegressor()
            model.fit(train_x,train_y)
            
            if global_step%10 == 0:
                f = open('./Pickle_all/PickleLocker_robotune_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(global_step) + '.p', 'wb')
                pickle.dump(model, f)  # 修改
                f.close()
        else:
            model = bagging(train_x=train_x.tolist(),train_y=train_y.tolist(),learning_model=model_name,file_name=filename,seed=seed,step=global_step,funcname="robotune")
            model.bagging_fit()
            model.save_model()
            
  
        y_min = min(train_y)

        x_max,x_maxs = gp_hedge(gains,y_min,model,file,global_step,Scaler_x,actions,xs)
        if model_name == "GP":
            # print("预测结果: ",[model.predict([x_maxs[i]]) for i in range(3)])
            for i in range(len(gains)):
                gains[i] -= model.predict([x_maxs[i]])
            # print("gains: ",gains)
        else:
            # print("预测结果: ",[model.predict([x_maxs[i]])[0][0] for i in range(3)])
            for i in range(len(gains)):
                gains[i] -= model.predict([x_maxs[i]])[0][0]
            # print("gains: ",gains)
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

    return np.array(xs),np.array(results), np.array(x_axis), best_result, best_loop, len(tuple_is)