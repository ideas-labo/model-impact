## Irace算法，采用paired T-test，然后有遗传的budget的思想

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import random
from math import log2
from scipy.stats import ttest_rel
from util.get_objective_model import get_objective_score_with_model
from util.read_model import read_model_class
from util.get_objective_model import get_path
from util.get_objective import get_objective_score_with_similarity


class one:
    def __init__(self, decision, sex, age=0) -> None:
        self.decision = decision 
        self.sex = sex
        self.age = age
        

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank

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

    


def change_p(p_list, round, parent, independent_set,N_iter,N):
    ## p_list是二维数组代表每个configuration的各自的概率
    # print(p_list, round, parent, independent_set,N_iter,N)
    for i in range(N):
        for j in range(len(p_list[i])):
            if parent[i] == independent_set[i][j]:
                delta_p = (round-1)/N_iter
            else:
                delta_p = 0
            p_list[i][j] = p_list[i][j]*(1-(round-1)/N_iter) + delta_p
    return p_list
    
def p_initial(independent_set):
    p_list = []
    for i in independent_set:
        p_list.append([1/len(i)]*len(i))
    return p_list

def p_reset(p_list):
    tmp_list = []
    for i in p_list:
        max_p = max(i)
        tmp = []
        for j in i:
            j = 0.9*j +0.1*max_p
            tmp.append(j)
        # 归一化
        SUM = sum(tmp)
        for i in range(len(tmp)):
            tmp[i] = tmp[i]/SUM
        tmp_list.append(tmp)
    # print(tmp_list)

    return tmp_list

def random_pick(some_list,probabilities):
    sum_p = sum(probabilities)
    probabilities = [i/sum_p for i in probabilities]
    # print(probabilities)
    if len(some_list) == 1:
        return some_list[0]
    x = random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item

def sample(p_list,independent_set,N):
    res = []
    for i in range(N):
        # print(independent_set,p_list)
        # print(independent_set[i],p_list[i])
        tmp = random_pick(independent_set[i], p_list[i])
        res.append(tmp)
        # print(tmp)
    return res

def pick_parent(elite):
    p_list = []
    n = len(elite)
    for i in range(len(elite)):
        p_tmp = (n-i)/(n*(n+1)/2)
        p_list.append(p_tmp)
    parent = random_pick(elite,p_list)
    return parent

def check_maxp(p_list,P_bound):
    tmp_list = []
    # print(p_list)
    # print(P_bound)
    for i in p_list:
        # if i == [0.0, 1.0]:
        #     i = 
        # elif i == [1.0, 0.0]
        tmp = []
        flag = 0
        SUM = 0
        for j in range(len(i)):
            if i[j] > P_bound:
                flag = 1
                tmp.append(P_bound)
            else:
                tmp.append(i[j])
                SUM += i[j]
        if flag == 1:
            for j in range(len(tmp)):
                if tmp[j]<P_bound-0.01:
                    # print(tmp[j])
                    # print(P_bound)
                    tmp[j] = tmp[j]/SUM*(1-P_bound)
        tmp_list.append(tmp)
    return tmp_list
        
def get_elite(candidates,file,model_name,N_min,T_each = 1,T_first = 5):
    scores = []
    for candidate in candidates:
        score = []
        for i in range(T_first):
            tmp = get_objective_score_similarly(candidate,file.dict_search,model_name)
            score.append(tmp)
        scores.append(score)
    
    # 按平均值排序
    one_with_score = []
    for j in range(len(candidates)):
        tmp_target = np.mean(scores[j])
        one_with_score.append((candidates[j],tmp_target))
    one_with_score = sorted(one_with_score,key= lambda x:x[1])
    print(one_with_score)
    # 只保留N_min个样本
    one_with_score = one_with_score[0:N_min]
    candidates = [i[0] for i in one_with_score]
    return candidates

class GotoFailedLabelException(Exception):
    pass

def run_irace(filename,model_name="GP",seed=1,maxlives=100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results,seed1,budget1
    budget1 = budget
    seed1 = seed
    maxlives = 2*maxlives
    model_predict = 0
    file_name = filename
    lives = maxlives
    max_lives = maxlives
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    B = 200
    file = get_data(filename)
    try:
        while 1:
            N_iter = int(2+log2(len(file.features)))
            N_min = N_iter
            elite= []
            N = len(file.features)
            P_bound = pow(0.2,1/N)
            p_list = p_initial(file.independent_set)
            for i in range(N_iter):
                num_candidate = B//(5+min(5,i))
                candidates = elite[:]
                # 生成candidate
                for j in range(num_candidate-len(elite)):
                    new_sample = sample(p_list,file.independent_set,N)
                    # print(new_sample)
                    # 参数相同的进行热重启
                    if new_sample in candidates:
                        p_list = p_reset(p_list)
                    else:
                        candidates.append(new_sample)
                # print(candidates)
                elite = get_elite(candidates,file,model_name,N_min)
                # 选出parent
                parent = pick_parent(elite)
                # print(parent)
                p_list = change_p(p_list,i,parent,file.independent_set,N_iter,N)
                # 检查边界是否超过限制
                p_list = check_maxp(p_list,P_bound)
                used_budget = len(set(map(tuple, xs)))
                # print(p_list)
                # print(used_budget)
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget
    return xs,x_result[1:],used_budget



             

            
