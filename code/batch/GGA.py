## GGA 遗传算法 with 两个性别 各自作用不同
import pandas as pd
import numpy as np
from random import shuffle
import random
import math
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
        self.independent_set = independent_set 
        self.features = features
        self.dict_search = dict_search


def get_data(filename, initial_size=5):
    """
    :param filename:
    :param Initial training size
    :return: file
    """
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col] 
    depcolumns = [col for col in pdcontent.columns if "$<" in col]       
    tmp_sortindepcolumns = []
    for i in range(len(indepcolumns)):
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[indepcolumns[i]]))))
    # print("去重排序：", tmp_sortindepcolumns)

    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1]) 
    ranks = {}

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



def crossover(a,b):
    child = []
    for i in range(len(a)):
        if random.random() > 0.5:
            child.append(a[i])
        else:
            child.append(b[i])
    return child

def mutation(child,independent_set, M=10):
    for i in range(len(child)):
        if random.random() < M/100:
            child[i] = random.sample(independent_set[i],1)[0]
    return child

class GotoFailedLabelException(Exception):
    pass


def run_gga(filename,model_name="GP",seed=1,maxlives= 100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results,seed1,budget1
    budget1 = budget
    seed1=seed
    maxlives = int(maxlives)
    model_predict = 0
    file_name = filename
    max_lives = maxlives
    lives = 100
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    A = 3
    M = 10
    X = 10
    MIN = 1e20
    initial_size = 50
    # filename = "./Data/Dune.csv"
    file = get_data(filename, initial_size)
    population = [one(i.decision,"C",random.sample([1,2,3],1)[0]) for i in file.training_set]
    for i in population[:initial_size//2]:
        i.sex = "N"
    # print(len(population))
    try:
        for i in range(1000):
            population = population[0:initial_size]
            one_with_score = []
            N_ones = []
            
            for j in population:
                if j.sex == "C":
                    tmp_target = get_objective_score_similarly(j.decision,file.dict_search, model_name)
                    if tmp_target < MIN:
                        MIN = tmp_target
                        print(MIN)
                    one_with_score.append((j,tmp_target))
                else:
                    N_ones.append(j.decision)
            one_with_score = sorted(one_with_score,key= lambda x:x[1])
            lens_C = len(one_with_score)
            lens_N = len(N_ones)
            # print(one_with_score)
            C_ones = [i[0].decision for i in one_with_score[0:math.ceil(X*lens_C/100)]]
            shuffle(N_ones)
            round = int(200/A*(lens_N)/100)
            lens = len(C_ones)

            for j in range(round):
                child = crossover(C_ones[j%lens],N_ones[j])
                child = mutation(child,file.independent_set)
                population.append(one(child,random.sample(["C","N"],1)[0]))
            for j in population:
                j.age += 1
                if j.age>3:
                    population.remove(j)      
            used_budget = len(set(map(tuple, xs)))
            print(used_budget)

            # print(min([get_objective_score(file.dict_search,j.decision) for j in population]))
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget
    return xs,x_result[1:],used_budget 

