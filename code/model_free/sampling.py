import random
## GA 遗传算法 with 两个性别 各自作用不同

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
from collections import defaultdict


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


class item():
    def __init__(self, id):
        self.id = id
        self.r = None
        self.d = None
        self.theta = None

def sway_continuous(pop):
    def cluster(items):
        # print(len(items))
        # add termination condition here
        print(len(items))
        if len(items) < 100:
            return items
            #  end at here
        scaler = MinMaxScaler()
        scaler.fit([i.id for i in items])
        for item in items:
            item.id = scaler.transform([item.id])[0]

        west, east, west_items, east_items = split_continuous(items)
        # return cluster(west_items)

        for item in west_items:
            item.id = scaler.inverse_transform([item.id])[0]

        for item in east_items:
            item.id = scaler.inverse_transform([item.id])[0]
        tmp_select = better(west.id,east.id)
        if tmp_select == 1:
            selected = east_items
        elif tmp_select == 2:
            selected = west_items
        else:
            selected = random.sample(west_items+east_items, len(items)//2)
            # return cluster(east_items) + cluster(west_items)
        # selected = west_items[:len(west_items)//2]+east_items[:len(east_items)//2]
        return cluster(selected)

    res = cluster(pop)

   
    return res


def sway_binary(pop):
    def cluster(items):
        # print(len(items))
        # add termination condition here
        if len(items) < 100:
            return items
            #  end at here

        wests, easts, west_items, east_items = split_binary(items)
        # return cluster(west_items)

        selected_all = []
        k = 0
        for west,east in zip(wests,easts):

            tmp_select = better(west.id,east.id)
            if tmp_select == 1:
                selected = east_items[k]
            elif tmp_select == 2:
                selected = west_items[k]
            else:
                selected = random.sample(west_items[k]+east_items[k], len(west_items[k]+east_items[k])//2)
            
            selected_all+=selected
            k += 1
            # return cluster(east_items) + cluster(west_items)
        # selected = west_items[:len(west_items)//2]+east_items[:len(east_items)//2]
        # print(selected_all)
        return cluster(selected_all)

    res = cluster(pop)

    # pdb.set_trace()
    return res


def better(item1, item2):
    global dict_search,modelname
    score1 = get_objective_score_similarly(item1,dict_search,modelname)
    score2 = get_objective_score_similarly(item2,dict_search,modelname)
    if score1 > score2:
        return 1
    elif score2 > score1:
        return 2
    else:
        return 3


import math
def hamming_distance(x, y):
    """计算两个等长01向量的汉明距离"""
    
    distance = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            distance += 1
    return distance

def num_1(x):
    num1=0
    for i in x:
        if i == 1:
            num1+=1
    return num1


def split_binary(items):
    # 随机选择一个item作为原点 
    rand = random.choice(items) 
    
    # 计算每个item的半径和距离
    print(len(items))
    for x in items:
        r = num_1(x.id) # x中“1”的个数
        d = hamming_distance(x.id, rand.id) # x与rand的汉明距离
        x.r = r  
        x.d = d

    # 按半径r分组
    on_same_r = defaultdict(list) 
    for x in items:
        on_same_r[x.r].append(x)
    # 在同一半径下,按d均匀分布角度
    for r, r_items in on_same_r.items():
        r_items.sort(key=lambda x: x.d)
        unit_angle = 2*math.pi/len(r_items)
        for i, x in enumerate(r_items):
            x.theta = i * unit_angle

    # 将空间等分为多个扇形区域
    num_areas = min(len(items[0].id),10)
    d_thresholds = np.linspace(0, max(x.d for x in items), num_areas)
    # print("d_thresholds:",d_thresholds)
    # 在每个扇形区域选代表性解
    west = [] 
    east = []
    west_items = []
    east_items = []
    for d_low, d_high in zip(d_thresholds, d_thresholds[1:]):
        west_item = []
        east_item = []
        for x in items:
            if d_low <= x.d <= d_high:
                # print(x.theta)
                if x.theta <= math.pi:
                    east_item.append(x)
                else:
                    west_item.append(x)
        if west_item:
            west_items.append(west_item)
            west.append(sorted(west_item,key=lambda x:x.theta)[-1])
        if east_item:
            east_items.append(east_item)
            east.append(sorted(east_item,key=lambda x:x.theta)[0])
        
    return west, east, west_items, east_items

class GotoFailedLabelException(Exception):
    pass


def euclidean_distance(x, y):
    """计算两个向量之间的欧氏距离"""
    
    distance = 0.0
    for xi, yi in zip(x, y):
        distance += (xi - yi)**2
        
    return math.sqrt(distance)

def split_continuous(items):

    # 随机选择一个item作为基准
    rand = random.choice(items)  
    
    # 找到与rand最远的east
    east = None
    max_dist = 0
    # print(rand)
    for x in items:
        dist = euclidean_distance(x.id, rand.id)
        if dist > max_dist:
            east = x
            max_dist = dist
            
    # 找到与east最远的west            
    west = None
    max_dist = 0
    for x in items:
        dist = euclidean_distance(x.id, east.id)
        if dist > max_dist:
            west = x
            max_dist = dist

    # 计算每个item到west和east的距离        
    for x in items:
        a = euclidean_distance(x.id, west.id)
        b = euclidean_distance(x.id, east.id)
        c = euclidean_distance(west.id, east.id)
        x.d = (a**2 + c**2 - b**2) / (2*c)
    
    # 按d排序        
    sorted_items = sorted(items, key=lambda x: x.d)
    
    # 划分两半
    mid = len(sorted_items) // 2
    west_items = sorted_items[:mid]
    east_items = sorted_items[mid:]
    
    return west, east, west_items, east_items


def run_sampling(filename,model_name="GP",seed=1,maxlives= 100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results
    global dict_search,modelname,seed1,budget1
    seed1 = seed
    budget1 = budget
    maxlives = int(maxlives)
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
    tmp_flag = 0
    try:
        for j in range(10):
            items = [item(i.decision) for i in file.all_set]
            random.seed((j+1)*seed)
            for i in file.independent_set:
                if len(i) > 2:
                    items = sway_continuous(items)
                    tmp_flag = 1
                    break
            if not tmp_flag:
                items = sway_binary(items)
            
            for i in items:
                get_objective_score_similarly(i.id,dict_search,model_name)
        
            xs = [tuple(x) for x in xs]
            used_budget = len(set(xs))
            print(used_budget)
    except GotoFailedLabelException:
    # print(xs,x_result[1:],used_budget)
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget 

    return xs,x_result[1:],used_budget 