import random
import time
import numpy as np
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

def get_objective_score(dict_search, best_solution):
    return dict_search.get(tuple(best_solution)) 

def get_objective_score_with_similarity(dict_search,best_solution):
    # print(dict_search)
    # print(best_solution)
    
    tmp = dict_search.get(tuple(best_solution))
    if tmp:
        return tmp,list(best_solution)
    else:
        scaler = MinMaxScaler()
        scaler.fit_transform([list(i) for i in list(dict_search.keys())])
        vectors = np.array(scaler.transform(list(dict_search.keys())))
        random.shuffle(vectors)
        kdtree = spatial.KDTree(vectors)
        query_vect = np.array(scaler.transform([best_solution])[0])
        # print(query_vect)
        # 搜索最近邻  
        dist, idx = kdtree.query(query_vect, k=1) 

        # 输出最近向量
        # print(vectors[idx])
        result = list(scaler.inverse_transform([vectors[idx]])[0])
        result = [round(i) for i in result]
        tmp_value = dict_search.get(tuple(result))
    if tmp_value:
        return tmp_value,result
    else:
        min_dist = 1e48
        list_dict = list(dict_search.keys())
        random.shuffle(list_dict)
        for i in list_dict:
            a = scaler.transform([list(i)])
            b = scaler.transform([best_solution])
            
            tmp = distance(a,b)

            if tmp < min_dist:
   
                min_dist = tmp
                tmp_value = dict_search.get(tuple(i))
                result = list(i)

        return tmp_value,result
    
    
def distance(p1,p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	dist_orig = np.sqrt(np.sum(np.square(p1-p2)))

	return dist_orig 
     

def distance(p1,p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	dist_orig = np.sum(np.square(p1-p2))

	return dist_orig