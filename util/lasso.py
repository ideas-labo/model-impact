#  使用lasso算法找出特征（ottertune）
# OtterTune - lasso.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from sklearn.linear_model import lasso_path
from abc import ABCMeta, abstractmethod
import numpy as np


class ModelBase(object, metaclass=ABCMeta):

    @abstractmethod
    def _reset(self):
        pass

def load_features (features_fileName):
    header = features_fileName.features
    features = [t.decision for t in features_fileName.all_set]
    target = [t.objective[-1] for t in features_fileName.all_set]
    return header , features , target


class LassoPath(ModelBase):
    """Lasso:
    Computes the Lasso path using Sklearn's lasso_path method.
    Attributes
    ----------
    feature_labels_ : array, [n_features]
                      Labels for each of the features in X.

    alphas_ : array, [n_alphas]
              The alphas along the path where models are computed. (These are
              the decreasing values of the penalty along the path).

    coefs_ : array, [n_outputs, n_features, n_alphas]
             Coefficients along the path.

    rankings_ : array, [n_features]
             The average ranking of each feature across all target values.
    """
    def __init__(self):
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def fit(self, X, y, feature_labels, estimator_params=None):
        """Computes the Lasso path using Sklearn's lasso_path method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (the independent variables).

        y : array-like, shape (n_samples, n_outputs)
            Training data (the output/target values).

        feature_labels : array-like, shape (n_features)
                         Labels for each of the features in X.

        estimator_params : dict, optional
                         The parameters to pass to Sklearn's Lasso estimator.

        Returns
        -------
        self
        """
        self._reset()
        if estimator_params is None:
            estimator_params = {}
        self.feature_labels_ = feature_labels

        alphas, coefs, _ = lasso_path(X, y, **estimator_params)
        self.alphas_ = alphas.copy()
        self.coefs_ = coefs.copy()

        # Rank the features in X by order of importance. This ranking is based
        # on how early a given features enter the regression (the earlier a
        # feature enters the regression, the MORE important it is).
        feature_rankings = [[] for _ in range(X.shape[1])]
        for target_coef_paths in self.coefs_:
            for i, feature_path in enumerate(target_coef_paths):
                entrance_step = 1
                for val_at_step in feature_path:
                    if val_at_step == 0:
                        entrance_step += 1
                    else:
                        break
                feature_rankings[i].append(entrance_step)
        self.rankings_ = np.array([np.mean(ranks) for ranks in feature_rankings])
        return self

    def get_ranked_features(self):
        if self.rankings_ is None:
            raise Exception("No lasso path has been fit yet!")

        rank_idxs = np.argsort(self.rankings_)
        return [self.feature_labels_[i] for i in rank_idxs]


# 传入：特征和文件
# 直接修改文件的内容
def clean_data(important_features,file):
    unimportant_features = []
    for i in file.features:
        if i not in important_features:
            unimportant_features.append(i)
    feature_sort = {x:i for i, x in enumerate(file.features)}  # 特征:特征编号
    # 转置
    feature_count = list(map(list, zip(*[t.decision for t in file.all_set])))
    # 得到出现最多的数
    feature_dict = {}
    for f in unimportant_features:
        tmp = feature_count[feature_sort[f]]
        x = max(set(tmp),key=tmp.count)
        feature_dict[f] = x
    # print(feature_dict.keys())
    # print(feature_dict.values())

    # 把file里面的元素删了
    for i in [file.all_set,file.training_set,file.testing_set]:
        for feature in unimportant_features:
            for j in i:
                if j.decision[feature_sort[feature]] != feature_dict[feature]:
                    i.remove(j)
    
    # 修改去重的自变量
    for feature in unimportant_features:
        file.independent_set[feature_sort[feature]] = [feature_dict[feature]]
    # print(file.independent_set)

def reset_data(file,initial_size):
    indexes = range(len(file.all_set))
    train_indexes, test_indexes = indexes[:initial_size],  indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes)== len(indexes)), "Something is wrong"
    file.training_set = [file.all_set[i] for i in train_indexes]
    file.testing_set = [file.all_set[i] for i in test_indexes]

