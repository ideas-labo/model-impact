from util import read_file
from util import lasso
from util import get_objective
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import heapq
import random
import pickle
from models import GPRNP
from util.bagging import bagging

def run_ottertune(initial_size, filename, model_name="GP", acqf_name = "UCB",budget=20, Beta=0.3,features_num=6,seed=0,maxlives=100):
    lives = maxlives
    test_size = 50
    file = read_file.get_data(filename)
    header, features, target = read_file.load_features(file)
    features_num = int(9/10*(len(header)))
    standardizer = StandardScaler()

    standardized_knob_matrix = standardizer.fit_transform(features)
    standardized_metric_matrix = standardizer.fit_transform([[i] for i in target])

    # run lasso algorithm
    lasso_model = lasso.LassoPath()
    lasso_model.fit(standardized_knob_matrix, standardized_metric_matrix, header)

    # consolidate categorical feature columns, and reset to original names
    encoded_knobs = lasso_model.get_ranked_features()[0:features_num]
    lasso.clean_data(encoded_knobs,file)
    lasso.reset_data(file,initial_size)

    ## 开始
    results = []
    x_axis = []
    xs = []
    tuple_is = [t.decision for t in file.training_set]

    memory = read_file.ReplayMemory()
    num_collections = initial_size  # 多少个已知量
    num_samples = test_size     # 多少个样本
    best_result = 1e20
    step = 0

    for i in range(num_collections):
        action = file.training_set[i]
        reward = action.objective[-1]
        memory.push(np.array(action.decision), np.array([reward]))
    # 主循环  # 关键是选了十个点
    
    while step+initial_size < budget:
        i = step
        step += 1
        # 随机寻找样本点
        X_samples = np.array([i.decision for i in random.sample(file.testing_set, num_samples)])

        # 对优秀的点周围增加探索
        # 将其动作乘以0.97加上0.01，并将其添加到X_samples中。这样做的目的是为了利用之前获得的信息，增加探索优秀动作的概率。
        if i >= 10:
            actions, rewards = memory.get_all()
            tuples = tuple(zip(actions, rewards))
            top10 = heapq.nlargest(10, tuples, key=lambda e: e[1])
            except_num = []
            for x in range(len(file.independent_set)):
                if len(file.independent_set[x]) == 1:
                    except_num.append(x)
                    
            for entry in top10:
                add_X_sample = " "
                count_i = 0
                while get_objective.get_objective_score(file.dict_search, add_X_sample) == None:
                    count_i += 1
                    randnum = random.randint(0,len(entry[0])-1)
                    while randnum in except_num: 
                        randnum = random.randint(0,len(entry[0])-1)
                    add_X_sample = entry[0].copy()
                    tmp_list = file.independent_set[randnum].copy()
                    tmp_list.remove(entry[0][randnum])
                    # print(tmp_list)
                    add_X_sample[randnum] = random.choice(tmp_list)
                    if count_i == 100:
                        break
                if count_i == 100:
                    continue
                X_samples = np.vstack((X_samples, add_X_sample))
        if model_name == "skip":
            model = GPRNP.GPRNP(length_scale=2.0,
                        magnitude=1.0,
                        max_train_size=2000,
                        batch_size=100,
                        debug=False,
                        )
            actions, rewards = memory.get_all()
            model.fit(np.array(actions), np.array(rewards))
            f = open('./Pickle_all/PickleLocker_ottertune_models'+ '/'+str(filename)[:-4] +'/'+str(model_name)+'_'+'seed'+str(seed) +'_'+'step'+str(step) + '.p', 'wb')
            if step%10 == 0:
                pickle.dump(model, f)  # 修改
                f.close()
            res = model.predict(X_samples)
            ypreds,sigmas = res.ypreds,res.sigmas
            if acqf_name == "UCB":
                loss = np.array([(ypred-Beta*sigma) for (ypred,sigma) in zip(ypreds,sigmas)])
                # loss = tf.subtract(np.array(res.ypreds), tf.multiply(Beta, np.array(res.sigmas)))
                # print(loss)
        else:
            actions, rewards = memory.get_all()
            model = bagging(train_x=actions,train_y=rewards,learning_model=model_name,file_name=filename,seed=seed,step=step,funcname="ottertune")
            model.bagging_fit()
            model.save_model()
            ypreds = []
            sigmas = []
            for j in range(len(X_samples)):
                ypred,sigma = model.predict([list(X_samples[j])])
                # print(ypred)
                ypreds.append(ypred.tolist()[0])
                sigmas.append(sigma)
            # print(ypreds)
            # print(sigmas)
            loss = np.array([(ypred-Beta*sigma) for (ypred,sigma) in zip(ypreds,sigmas)])
            # loss = tf.subtract(np.array(ypreds).astype(np.float32), tf.multiply(Beta, np.array(sigmas).astype(np.float32)))
            # print("loss: ",loss)

  
        best_config_idx = np.argmin(loss)  
        best_config = X_samples[best_config_idx]
        # sorted_merge = sorted(zip(list(X_samples),list(loss)), key=lambda x: x[1], reverse=False)
        # print(sorted_merge)
        # for s,m in sorted_merge:

        #     if list(s) in actions:
        #         continue
        #     else:
        #         best_config = s
        #         break

        # print(best_config)
        reward,tuple_i = get_objective.get_objective_score_with_similarity(file.dict_search, list(best_config))
        memory.push(best_config, np.array([reward]))
        print('loop: ', i, 'reward: ', reward)
        results.append(reward)
        x_axis.append(i+1)
        xs.append(tuple_i)

        if tuple_i in tuple_is:
            budget += 1
        else:
            tuple_is.append(tuple_i)


        if reward < best_result:
            best_result = reward
            best_loop = i
            lives = maxlives
        lives -= 1
        if lives == 0:
            break
    return np.array(xs), np.array(results), np.array(x_axis), best_result, best_loop, len(tuple_is)