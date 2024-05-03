from __future__ import division
import pandas as pd
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from util.bagging import bagging
import pickle
# 定义变量
class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank

# 得到数据
# Return：训练集和测试集
def get_data(filename, initial_size):
    """
    :param filename:
    :param Initial training size
    :return: Training and Testing
    """
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]  # 自变量
    depcolumns = [col for col in pdcontent.columns if "$<" in col]        # 因变量
    
    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])  # 按目标从小到大排序
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):  # 目标转化为list再去重，再排序
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

    shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:initial_size],  indexes[initial_size:]
    assert(len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]

# Acquire Function: CART算法来预测测试集的效果
# Return: 返回一test的评价分数list
# 在该步骤加入了新的点,来训练CART模型
def get_best_configuration_id(train, test, model_name,filename,seed,step):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    if model_name == "DT":
        model = DecisionTreeRegressor()
        model.fit(train_independent, train_dependent)
        
        if step%10 == 0:
            f = open('./new_pickle/cpz-work/Pickle_all/PickleLocker_flash_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step) + '.p', 'wb')
            pickle.dump(model, f)  # 修改
            f.close()
    else:
        model = bagging(train_x=train_independent,train_y=train_dependent,learning_model=model_name,file_name=filename,seed=seed,step=step,funcname="flash")
        model.bagging_fit(estimators_num = 0)
        model.save_model()

    predicted = [model.predict([test]) for test in test_independent]

    predicted_id = [[t.id,p] for t,p in zip(test, predicted)]
  
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # Find index of the best predicted configuration

    best_index = predicted_sorted[0][0]
    return best_index

# 主函数 返回训练集和测试集 进行了训练集运算和扩充
def run_active_learning(filename, initial_size, model_name,seed,budget,max_lives=10):
    steps = 1
    lives = max_lives
    training_set, testing_set = get_data(filename, initial_size)
    dataset_size = len(training_set) + len(testing_set)
    x_axis = []
    xs = []
    results = []
    best_loop = 0
    best_result = 0
    while steps+initial_size <= budget:
        best_id = get_best_configuration_id(training_set, testing_set, model_name,filename,seed,steps)

        best_solution = [t for t in testing_set if t.id == best_id][-1]  # 得到测试集中最好的目标值
        results.append(best_solution.objective[-1])

        x_axis.append(steps)
        xs.append(best_solution.decision)
 
        list_of_all_solutions = [t.objective[-1] for t in training_set]  # 训练集里的所有值

        if best_solution.objective[-1] < min(list_of_all_solutions):
            lives = max_lives
            best_loop = steps
            best_result = best_solution.objective[-1]
        else:
            lives -= 1
        
        training_set.append(best_solution)
        # find index of the best_index
        best_index = [i for i in range(len(testing_set)) if testing_set[i].id == best_id]
        assert(len(best_index) == 1), "Something is wrong"
        best_index = best_index[-1]
        del testing_set[best_index]
        assert(len(training_set) + len(testing_set) == dataset_size), "Something is wrong"
        if lives == 0:
            break
        steps += 1
       

    return np.array(xs), np.array(results), np.array(x_axis),best_loop,best_result,training_set, testing_set

# 打包
def run_flash(initial_size,filename,model_name="DT",seed=0,budget=20, maxlives=100):
    xs,results,x_axis,best_loop,best_result,training_set, testing_set= run_active_learning(filename, initial_size,model_name,seed,budget,max_lives=maxlives) 
    global_min = min([t.objective[-1] for t in training_set + testing_set])
    best_training_solution = [ tt.rank for tt in training_set if min([t.objective[-1] for t in training_set]) == tt.objective[-1]]
    best_solution = [tt.rank for tt in training_set + testing_set if tt.objective[-1] == global_min]
    
    # print ("差距:", min(best_training_solution) - min(best_solution)), len(training_set), " | ",
    return xs,results,x_axis,best_loop,best_result,len(results)+budget
# if __name__ == "__main__":
#     filenames = ["./Data/"+f for f in listdir("./Data")]
#     initial_size = 20
#     evals_dict = {}
#     rank_diffs_dict = {}
#     stats_dict = {}
#     for filename in filenames:
#         evals_dict[filename] = []
#         rank_diffs_dict[filename] = []
#         stats_dict[filename] = {}
#         rank_diffs = []
#         evals = []
#         print (filename)
#         for _ in range(20):
#             temp1, temp2 = wrapper_run_active_learning(filename, initial_size)
#             rank_diffs.append(temp1)
#             evals.append(temp2)
#         # print
#         evals_dict[filename] = evals
#         rank_diffs_dict[filename] = rank_diffs
#         stats_dict[filename]["mean_rank_diff"] = np.mean(rank_diffs)
#         stats_dict[filename]["std_rank_diff"] = np.std(rank_diffs)
#         stats_dict[filename]["mean_evals"] = np.mean(evals)
#         stats_dict[filename]["std_evals"] = np.std(evals)

#     # import pickle
#     # pickle.dump(evals_dict, open("./PickleLocker/ActiveLearning_Evals.p", "wr"))
#     # pickle.dump(rank_diffs_dict, open("./PickleLocker/ActiveLearning_Rank_Diff.p", "wr"))
#     # pickle.dump(stats_dict, open("./PickleLocker/ActiveLearning_Stats.p", "wr"))

