
import pandas as pd
import random
from util.read_model import read_model_class
from util.get_objective_model import get_path
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

def evolve(conf_best, confs_accepted,f_len,independent_set):

    conf_children = [] 

    for conf in confs_accepted:
        crossover_params_index = random_select_params(f_len, ratio=0.5) 
        # print(crossover_params_index)
        child = crossover(conf_best, conf, crossover_params_index)
        conf_children.append(child)

    mutation_params = random_select_params(f_len, ratio=0.12)
    for child in conf_children:
        child = mutate(child, mutation_params,independent_set)

    return conf_children
def random_select_params(f_len,ratio):
    n = int(f_len*ratio)
    if n == 0:
        n = 1
    return random.sample(range(f_len),n)

def random_sample(configs, n):
    return random.sample(configs, n) 

def evaluate(best_solution):
    global dict_search,modelname
    return get_objective_score_similarly(best_solution,dict_search,modelname)

def accept(perf_best, perf):
    try:
        p = min(1,perf_best/perf)
    except Exception:
        p = 1
    p_random = random.random()
    if  p_random <= p:
        # print(perf_best,perf)
        # print(p_random,p)
        return True
    else:
        return False

def crossover(conf1, conf2, params):
    child = conf1.copy()
    for param in params:
        child[param] = conf2[param] 
    return child

def mutate(conf, params,independent_set):
    child = conf.copy()
    for param in params:
        child[param] = random.sample(independent_set[param],1)[0]
    return child

class GotoFailedLabelException(Exception):
    pass

def run_conex(filename,model_name="GP",seed=1,maxlives= 100,budget=100):
    global flag,xs,x_result,max_lives,lives,file_name,model_predict,tmp_results
    global dict_search,modelname,seed1,budget1
    budget1 = budget
    seed1 = seed
    maxlives = int(maxlives*1.5)
    model_predict = 0
    file_name = filename
    max_lives = maxlives
    lives = 100
    flag = 0
    xs = []
    x_result = [1e24]
    tmp_results = [1e24]
    initial_size = 5
    file = get_data(filename, initial_size)
    dict_search = file.dict_search
    modelname  = model_name
    last_budget = 0
    # feature_len = [len(tmp_sortindepcolumn) for tmp_sortindepcolumn in file.tmp_sortindepcolumns]
    conf_parents_all = [i.decision for i in file.testing_set]
    try:
        for j in range(10):
            conf_best = 0
            #   perf_best = evaluate(conf_seed)
            perf_best = 1e36
            f_len = len(file.independent_set)
            n = min(4*f_len,60)
            conf_parents = random.sample(conf_parents_all,n)
            #   conf_parents = random_sample(configs, n) 
            
            for i in range(30):
                print("conex:",i+1)
                confs_accepted = []

                # 评估每个父配置
                for conf in conf_parents:
                    # print
                    perf = evaluate(conf)
                    if accept(perf_best, perf):
                        confs_accepted.append(conf)
                    if perf < perf_best:
                        conf_best = conf
                        perf_best = perf
                conf_parents = evolve(conf_best, confs_accepted,f_len,file.independent_set)
                # print(conf_parents)
                xs = [tuple(x) for x in xs]
                used_budget = len(set(xs))
                if last_budget == used_budget:
                    break
                last_budget = used_budget
                print(used_budget)
            # break
    except GotoFailedLabelException:
        xs = [tuple(x) for x in xs]
        used_budget = len(set(xs))
        return xs,x_result[1:],used_budget 
    return xs,x_result[1:],used_budget 

