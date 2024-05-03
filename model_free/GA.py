

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import random
import math
from util.get_objective_model import get_objective_score_with_model
from util.read_model import read_model_class
from util.get_objective_model import get_path
from scipy import spatial
from util.get_objective import get_objective_score_with_similarity



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





# 两个点交叉  
def crossover(ind1, ind2):
    tmp1 ,tmp2 = [],[]
    for i in range(len(ind1)):
        if random.random() < 0.8:
            tmp1.append(ind1[i])
            tmp2.append(ind2[i])
        else:
            tmp1.append(ind2[i])
            tmp2.append(ind1[i])
    return tmp1,tmp2

# 随机点位变异
def mutation(ind,independent_set):
    for i in range(len(ind)):
        if random.random() < 2/len(independent_set):
            ind[i] = random.sample(independent_set[i],1)[0]
    return ind

# 评价函数
def evaluate(best_solution):
    global dict_search,modelname
    return get_objective_score_similarly(best_solution,dict_search,modelname)

def tournament_selection(pop, pop_size):
    next_gen = []
    for j in range(pop_size):
        # 随机选择k个个体进行比赛
        k = 3
        competitors = random.sample(pop, k)
        
        #选择fitness最高的个体作为胜者 
        winner = min(competitors, key=fitness)
        # print(j)
        # 将胜者添加到下一代
        next_gen.append(winner)

    return next_gen

def fitness(best_solution):
    global dict_search,modelname
    return get_objective_score_similarly(best_solution,dict_search,modelname)

class GotoFailedLabelException(Exception):
    pass

def run_ga(filename,model_name="GP",seed=1,maxlives= 100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results
    global dict_search,seed1,modelname,budget1
    budget1 = budget
    seed1=seed
    maxlives = int(maxlives*3)
    model_predict = 0
    file_name = filename
    max_lives = maxlives
    lives = 100
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    initial_size = 100
    file = get_data(filename, initial_size)
    dict_search = file.dict_search
    modelname  = model_name
    # feature_len = [len(tmp_sortindepcolumn) for tmp_sortindepcolumn in file.tmp_sortindepcolumns]
    pop_size = 100
    pop = [i.decision for i in file.training_set]
    try:
        for count in range(100):
            print("GA:",count+1)
            pop = tournament_selection(pop, pop_size)
            # print(len(pop))
            new_pop = []
    
            for i in range(pop_size//2):
                c1, c2 = crossover(pop[i], pop[99-i])
                new_pop.append(c1) 
                new_pop.append(c2)
            # print(new_pop)
            random.shuffle(new_pop)
            pop = new_pop
            # print(len(pop))
            m_num = int(0.1*len(pop))
            numbers = np.random.choice(100, m_num)
    
            for number in numbers:
                tmp = pop[number]
                tmp = mutation(tmp,file.independent_set)
                pop[number] = tmp

            # best = max(pop, key=evaluate)
            # print(len(xs))
            # xs = [tuple(x) for x in xs]
            # used_budget = len(set(xs))
            # print(xs,x_result[1:],used_budget)
            # if used_budget >= budget:
            #     return xs,x_result[1:],used_budget  
            # if flag == 1:
            #     return xs,x_result[1:],used_budget 
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget

    return xs,x_result[1:],used_budget 

