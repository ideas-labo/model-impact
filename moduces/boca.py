
from os import listdir
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from scipy.stats import norm
import math
from itertools import product
from util import get_objective
from util import read_file
from util.bagging import bagging
import pickle


def model_randomforest(train_independent, train_dependent):

    model = RandomForestRegressor()
    model.fit(train_independent, train_dependent)

    return model

# 得到EI：用于acquire function
def ei(m,s,eta):
    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    if s==0:
        return 0
    else:
        f = calculate_f()

    return f
def get_ei(pred, eta):  #eta最小值
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)

    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()

    return f

import math

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

# 返回：测试序列，模型

def get_training_sequence_by_BOCA(fnum, rnum, training_indep, training_dep, sort_indepcolumns):
    model = model_randomforest(training_indep, training_dep)
    ######## 替换模型修改这里 ############# 
    features = model.feature_importances_
    feature_sort = [[i, x] for i, x in enumerate(features)]  # 特征及特征编号
    feature_selected = sorted(feature_sort, key=lambda x: x[1], reverse=True)[
        :fnum]  # 选出来的重要特征
    feature_not_selected = sorted(
        feature_sort, key=lambda x: x[1], reverse=True)[fnum:]
    comb, comb1 = [], []
    inc_important = []
    inc_unimportant = []

    # 生成重要的特征
    tmp_important = []
    for i in feature_selected:
        tmp_important.append(sort_indepcolumns[i[0]])
    # 含有的特征数量太多了，抽样减少一部分
    for i in range(len(tmp_important)):
        if len(tmp_important[i])>10:
            # print(tmp_important[i])
            tmp_important[i] = fixed_interval_sample(tmp_important[i],10)
    
    for i in product(*[i for i in tmp_important]):
        comb.append(i)
    
    # for i in range(2 ** fnum):
    #     comb = bin(i).replace('0b', '')
    #     comb = '0' * (fnum - len(comb)) + comb
    #     # 生成 [重要特征序号，它的值]
    for i in comb:
        tmp = []
        for k, s in enumerate(i):
            tmp.append([feature_selected[k][0], s])
        inc_important.append(tmp)
    # print('tmp_important',len(inc_important))
    # 生成不重要的特征
    tmp_not_important = []
    # print(sort_indepcolumns)
    for i in feature_not_selected:
        tmp_not_important.append(sort_indepcolumns[i[0]])
    # print(tmp_not_important)
    # def gen_comb1():
    #     for i in product(*[i for i in tmp_not_important]):
    #         yield i
    # comb1 = random.sample(gen_comb1(), int(rnum))
    # for i in product(*[i for i in tmp_not_important]):
    #     comb1.append(i)
    # comb1 = random.sample(comb1, int(rnum))
    
    for _ in range(max(int(rnum),1)):
        single = []
        for i in range(len(tmp_not_important)):
            single.append(random.choice(tmp_not_important[i]))
        comb1.append(single)
    for i in comb1:
        tmp = []
        for k, s in enumerate(i):
            tmp.append([feature_not_selected[k][0], s])
        inc_unimportant.append(tmp)
    # print('tmp_umimportant',len(inc_unimportant))
    # for i in range(int(rnum)):
    #     tmp = [[feature_not_selected[i][0], round(random.uniform(lower_bound[i] * 1.0, upper_bound[i] * 1.0))]
    #            for i in range(len(feature_not_selected))]
    #     inc_unimportant.append(tmp)

    inc = []
    for i in inc_important:
        for j in inc_unimportant:
            tmp = []
            tmp = i + j
            tmp = sorted(tmp, key=lambda x: x[0], reverse=False)
            inc.append(tmp)
    # print(inc[0],inc[1])
    # print("Have finished find test sequence")
    return inc, model

# 返回：下一个抽样点

def get_best_configuration_id_BOCA(training_indep, training_dep, all_indep, fnum, rnum, eta, sort_indepcolumns,model_name,filename,seed,step):
    test_sequence, model = get_training_sequence_by_BOCA(
        fnum, rnum, training_indep, training_dep, sort_indepcolumns)
    # print("Test consequence len: ", len(test_sequence))
    # f = open('./Pickle_all/PickleLocker_boca_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step)+'_indeps' + '.p', 'wb')
    # if step%10 == 0:
    #     pickle.dump(training_indep, f)
    # f = open('./Pickle_all/PickleLocker_boca_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step)+'_deps' + '.p', 'wb')
    # if step%10 == 0:
    #     pickle.dump(training_dep, f)
    # print("完成pickle")
    merged_ei = []
    if model_name == "RF11":
        estimators = model.estimators_

        independent = []
        for i in range(len(test_sequence)):
            pred = []
            independent = [test_sequence[i][j][1]
                        for j in range(len(test_sequence[i]))]
            # print(independent)
            for e in estimators:
                pred.append(e.predict([independent]))
            # print(pred)
            value = get_ei(pred, eta)
            # print(value)
            ei_zip = [independent, value]
            merged_ei.append(ei_zip)
            
            if step%10 == 0:
                f = open('./Pickle_all/PickleLocker_boca_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step) + '.p', 'wb')
                pickle.dump(model, f)  # 修改
                f.close()
    else:
        model = bagging(training_indep,training_dep,learning_model=model_name,file_name=filename,seed=seed,step=step,funcname="boca")
        model.bagging_fit()
        # print(len(test_sequence))
        for i in range(len(test_sequence)):

            independent = [test_sequence[i][j][1] for j in range(len(test_sequence[i]))]
            # print("---",independent)
            pred,std = model.predict(x=[independent])
            # print(pred,std)
            pred = pred[0]
            value = ei(pred,std,eta)
            # print(value)
            ei_zip = [independent, value]
            # print("---",ei_zip)
            merged_ei.append(ei_zip)
        model.save_model()
    # print(merged_ei)
    # import joblib
    # joblib.dump(model,"./PickleLocker_BOCA_model/Apache_AllNumeric/DaL_seed1.m")
    sort_merged_ei = sorted(merged_ei, key=lambda x: x[1], reverse=True)
    # print("sort:",sort_merged_ei)
    return (sort_merged_ei[0][0], model)
    
    print("Not found, please correct the parameters: fnum, rnum")
            
# 返回：对应decision的object，和decision（格式化之后）




def run_boca(filename, initial_size, maxlives, budget, fnum, rnum0,model_name="RF",seed=0):
    # fnum定义的重要特征个数 # rnum0衰减函数初始值，即一开始要采样的非重要特征的个数
    
    steps = 1
    lives = maxlives
    scale = float(10)
    decay = float(0.5)
    sigma = -scale ** 2 / (2 * math.log(decay))
    offset = float(20)

    file = read_file.get_data(filename, initial_size)
    header,_,_=read_file.load_features(file)
    lens = len(header)
    
    fnum = min(max(int(1/2*lens),1),fnum)
    rnum0 = min(2**(lens-fnum-1),rnum0)
    print("fnum:",fnum)
    print("rnum0",rnum0)
    training_dep = [t.objective[-1] for t in file.training_set]
    all_indep = [t.decision for t in file.all_set]

    # 初始化最优值
    result = 1e20

    for x in training_dep:
        if result > x:
            result = x

    results = []
    x_axis = []
    xs = []
    

    training_indep = [t.decision for t in file.training_set]
    training_dep = [t.objective[-1] for t in file.training_set]
    tuple_is = training_indep[:]

    best_loop = 1
    print("进入循环")
    while steps+initial_size <= budget:
        # print("len of training independent: ", len(training_indep))
        rnum = rnum0 * \
            math.exp(-max(0, len(training_indep) - offset)
                     ** 2 / (2 * sigma ** 2))
        (best_solution, model) = get_best_configuration_id_BOCA(
            training_indep, training_dep, all_indep, fnum, rnum, result, file.independent_set,model_name,filename,seed,steps)

        
        best_result,tuple_i = get_objective.get_objective_score_with_similarity(
            file.dict_search, best_solution)
        # print(best_result)
        training_indep.append(best_solution)
        training_dep.append(best_result)
        x_axis.append(steps)
        results.append(best_result)
        xs.append(tuple_i)

        if tuple_i in tuple_is:
            budget += 1
        else:
            tuple_is.append(tuple_i)

        if best_result < result:
            result = best_result
            lives = maxlives
            best_loop = steps
        else:
            lives -= 1
      
        if lives == 0:
            break
        print('loop: ', steps, 'now best:',result,'reward: ', best_result)
        steps += 1
    return np.array(xs), np.array(results), np.array(x_axis), result, best_loop,len(tuple_is)


# if __name__ == "__main__":
#     filenames = ["./Data/"+f for f in listdir("./Data")]
#     # for filename in filenames:
#     #     run_BOCA(filename, initial_size=20,
#     #              maxlives=10, budget=40, fnum=4, rnum0=2**4)
#     run_boca(filename="./Data_small/Apache_AllNumeric.csv",
#              initial_size=20, maxlives=10, budget=40, fnum=2, rnum0=2**5)
