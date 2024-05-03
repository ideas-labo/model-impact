## ParamILSParamILS
## 没有封盖，一个benchmark，使用最简单的basicILS

import pandas as pd
import numpy as np
import random
from random import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from util.get_objective_model import get_objective_score_with_model
from util.read_model import read_model_class
from util.get_objective_model import get_path
from util.get_objective import get_objective_score_with_similarity

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank

# 定义文件

class file_data:
    def __init__(self, name, training_set, testing_set, all_set, independent_set, features, dict_search):
        self.name = name
        self.training_set = training_set
        self.testing_set = testing_set
        self.all_set = all_set
        self.independent_set = independent_set  # 自变量的值
        self.features = features
        self.dict_search = dict_search

# 得到数据
# 返回：file_data

def get_data(filename, initial_size=5):
    """
    :param filename:
    :param Initial training size
    :return: file
    """
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]  # 自变量
    depcolumns = [col for col in pdcontent.columns if "$<" in col]        # 因变量

    # 对自变量列进行排序和去重
    tmp_sortindepcolumns = []
    for i in range(len(indepcolumns)):
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[indepcolumns[i]]))))
    print("去重排序：", tmp_sortindepcolumns)

    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])  # 按目标从小到大排序
    ranks = {}
    # 目标转化为list再去重，再排序
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i

    content = list()
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
        )
        )
    dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))
    random.shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:
                                          initial_size],  indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes)
            == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    file = file_data(filename, train_set, test_set,
                     content, tmp_sortindepcolumns, indepcolumns, dict_search)
    return file

def get_objective_score_similarly(best_solution,dict_search,model_name):
    global flag,xs,x_result,lives,max_lives,file_name,model_predict,tmp_results,seed1,budget1

    
    if model_name == "real":
        
        tmp_result,x = get_objective_score_with_similarity(dict_search,best_solution)
        xs.append(list(x))
        if min(x_result)>tmp_result:
            lives = max_lives
        else:
            lives -= 1
        x_result.append(tmp_result)
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        if used_budget >= budget1:
            raise GotoFailedLabelException
        if lives == 0:
            flag = 1
            raise GotoFailedLabelException
        return tmp_result
    else:
        
        if model_predict == 0:
            save_path = get_path(learning_model=model_name,file=file_name,bayes_models="None",seed=seed1)
            model_predict = read_model_class(model_name,save_path,"None")
            tmp_result = model_predict.predict(best_solution)
        else:
            tmp_result = model_predict.predict(best_solution)

            
        if min(tmp_results)>tmp_result:
            lives = max_lives
        else:
            lives -= 1
        tmp_result_1,x = get_objective_score_with_similarity(dict_search,best_solution)
        xs.append(list(x))
        x_result.append(tmp_result_1)
        tmp_results.append(tmp_result)
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        if used_budget >= budget1:
            raise GotoFailedLabelException
        if lives == 0:
            flag = 1
            raise GotoFailedLabelException
        return tmp_result


def fixed_interval_sample(lst, n, keep_ends=True):
    # 计算间隔
    interval = (len(lst) / (n-1))  
    # print(interval)
    # 抽样索引
    indices = [0] + [int(i*interval) for i in range(1, n-1)]
    # print(indices)
    if keep_ends:
        indices.append(-1)
    
    # 抽样并保留原序    
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

def iterative_first_improvement(configuration,file,model_name):
    while 1:
        tmp_configuration = configuration
        tmp = get_objective_score_similarly(tmp_configuration,file.dict_search,model_name)
        for i in find_neighbourhood(tmp_configuration,file):
            if get_objective_score_similarly(i,file.dict_search,model_name) < tmp:
                configuration = i
                # print(get_objective_score_similarly(i))
                break
        if tmp_configuration == configuration:
            break
    return configuration

class GotoFailedLabelException(Exception):
    pass

def run_paramILS(filename,model_name="GP",r=10,s=3,p_restart=0.01,seed=1,maxlives=100,budget=100):

    # r = 6 ## 10  ## 初始采样点，在其中选择最好的点，然后进行一次提升迭代
    # s 选择多少个邻居来脱离局部最优点 为文章的选择 有些任意性，本来后面paramILS（N）表示有N个benchmark。
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results,seed1,budget1
    budget1 = budget
    seed1 = seed
    maxlives = int(maxlives*2)
    model_predict = 0
    file_name = filename
    max_lives =maxlives
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    lives = maxlives
    loop = 0
    file = get_data(filename, initial_size=r)
    result = 1e20
    for i in file.training_set:
        if i.objective[-1] < result:
            result = i.objective[-1]
            init_configuration = i.decision
    # import pickle
    # import os
    # dict_result = {}
    # for i in file.training_set:
    #     dict_result[tuple(i.decision)] = i.objective[-1]
    # with open('./initial_set/'+"paramILS"+'_'+os.path.basename(filename)+'_'+str(seed)+'.pkl', 'wb') as f:
    #     pickle.dump(dict_result, f)   
    # raise KeyError
    init_configuration = iterative_first_improvement(init_configuration,file,model_name)
    print("第一次的结果: ", get_objective_score_similarly(init_configuration,file.dict_search,model_name))
    ## 原文未说进行多少次
    try:
        while loop <= 50: #2
            tmp_configuration = init_configuration
            for _ in range(s):
                tmp_configuration = random.sample(find_neighbourhood(tmp_configuration,file),1)[0]
                tmp_configuration = iterative_first_improvement(tmp_configuration,file,model_name=model_name)
                tmp = get_objective_score_similarly(init_configuration,file.dict_search,model_name)
                if get_objective_score_similarly(tmp_configuration,file.dict_search,model_name) < tmp:
                    init_configuration = tmp_configuration
                    print("扰动找到的结果： ",tmp)
            random_tmp = random.random()
            if random_tmp <= p_restart:
                init_configuration = random.sample([i.decision for i in file.all_set],1)[0]
            loop += 1
            used_budget = len(set(map(tuple, xs)))
            print(used_budget)
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget
    return xs,x_result[1:],used_budget





