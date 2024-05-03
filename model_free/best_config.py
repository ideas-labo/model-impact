## bestconfig使用分割循环查找

import pandas as pd
import numpy as np
from doepy import build
from random import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from util.get_objective_model import get_path
import random
from util.read_model import read_model_class

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
    xs.append(best_solution)
    
    if model_name == "real":
        tmp_result = get_objective_score(best_solution,dict_search)
        
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
            # print(best_solution)
            tmp_result = model_predict.predict(best_solution)
            # print("cao",model_predict.predict([0,1,0,0,1,3,0.6,23,3,40,0,4,250,23,40,0,1.4]))
            # print("cao",model_predict.predict([0,0,0,0,14,8,1,31,8,161,90,85,877,3.37E+14,14,1042,6]))
            # print(tmp_result)
        else:
            tmp_result = model_predict.predict(best_solution)

            
        
        if min(tmp_results)>tmp_result:
            lives = max_lives
        else:
            lives -= 1
        x_result.append(get_objective_score(best_solution,dict_search))
        tmp_results.append(tmp_result)
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        if used_budget >= budget1:
            raise GotoFailedLabelException
        if lives == 0:
            flag = 1
            raise GotoFailedLabelException
        return tmp_result




class GotoFailedLabelException(Exception):
    pass

def get_objective_score(best_solution,dict_search):
    tmp = dict_search.get(tuple(best_solution))
    if tmp:
        return tmp
    else:
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

def distance(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist_orig = np.sqrt(np.sum(np.square(p1-p2)))

    return dist_orig

def Latin_sample(file, bounds, num_samples = 20):
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

    # 当数据中存在只有一个的set需要进行该操作
    while [0.0] in bounds:
        bounds = [[0.0,1.0] if i == [0.0] else i for i in bounds]
    while [1.0] in bounds:
        bounds = [[0.0,1.0] if i == [1.0] else i for i in bounds]
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

# 生成bound边界
def make_bound(sample, best_now):
    bound = []
    # print(sample)
    for i in range(len(sample[0])):
        max_tmp = 1e20
        min_tmp = -1e20
        min_bound = -1
        max_bound = -1
        for j in range(len(sample)):
            # if best_now[i] == 0.0:
            #     min_bound = 0.0
            #     max_bound = 1.0
            # if best_now[i] == 1.0:
            #     max_bound = 1.0
            #     min_bound = 0.0
            if sample[j][i] - best_now[i] > 0 and sample[j][i] - best_now[i] < max_tmp:
                max_tmp = sample[j][i] - best_now[i]
                max_bound = sample[j][i]
            elif sample[j][i] - best_now[i] < 0 and sample[j][i] - best_now[i] > min_tmp:
                min_tmp = sample[j][i] - best_now[i]
                min_bound = sample[j][i]
        if min_bound == -1:
            min_bound = best_now[i]
        if max_bound == -1:
            max_bound = best_now[i]
        if min_bound == max_bound:
            min_bound,max_bound = 0.0,1.0
        # print([min_bound,max_bound])
        # print(best_now)
        bound.append([min_bound, max_bound])
    return bound
            

def search(sample,file,model_name):
    best_score = 1e32
    for i in range(len(sample)):

        tmp_score = get_objective_score_similarly(sample[i],file.dict_search,model_name)
        # print(tmp_score)
        if  tmp_score < best_score:
            best_score = tmp_score
            best_result = sample[i]
    return best_result,best_score

# 总次数：总的times*number
# 要保证每个loop够大在，再保证轮数多一点，避免陷入局部最优

def run_best_config(filename,model_name="GP",seed=1,maxlives=100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results,seed1,budget1
    budget1 = budget
    seed1=seed
    model_predict = 0
    file_name = filename
    max_lives = maxlives
    lives = maxlives
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    file = get_data(filename)
    number = 8
    loop = 100
    bound = file.independent_set
    result = []
    try:
        for i in range(loop):
            best_score = 1e32
            bound = file.independent_set
            k = 0
            while 1:
                # print(xs)
                sample = Latin_sample(file, bound, num_samples=number)
                new_result, new_score = search(sample,file,model_name)
                if new_score >= best_score:
                    break
                else:
                    best_score = new_score
                k += 1
                print("loop: ",i+1,"times: ",k,"score: ",best_score)
                bound = make_bound(sample, new_result)
            print("loop: ",i+1,"score: ",best_score)
            result.append([new_result,best_score])
        # print(result)
            xs = [tuple(x) for x in xs]
            used_budget = len(set(xs))
 
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget
    return xs,x_result[1:],used_budget
