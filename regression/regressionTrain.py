#!/usr/bin/env python
# -*- coding:utf-8 -*-

import joblib
from classification.Train1 import ClassificationTrainFromSklearn, ClassificationTrain
from sklearn.base import RegressorMixin
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PolynomialRegression(PolynomialFeatures, LinearRegression):
    """
    多项式线性回归模型
    """
    __PF_params = ["degree", "interaction_only", "include_bias", "order"]
    __LR_params = ["normalize", "copy_X", "n_jobs"]

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order='C',
                 normalize=False, copy_X=True, n_jobs=None):
        PolynomialFeatures.__init__(self, degree=degree, interaction_only=interaction_only, include_bias=include_bias,
                                    order=order)
        LinearRegression.__init__(self, fit_intercept=False, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

    def get_params(self, deep=True):
        return PolynomialFeatures.get_params(self, deep = deep)

    def set_params(self, **params):
        dict1 = {k: v for k, v in params.items() if k in self.__PF_params}
        dict2 = {k: v for k, v in params.items() if k in self.__LR_params}
        PolynomialFeatures.set_params(self, **dict1)
        LinearRegression.set_params(self, **dict2)

    def fit_transform(self, X, y=None, **fit_params):
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return super(PolynomialRegression, self).fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return super(PolynomialRegression, self).fit(X, y, **fit_params).transform(X)

    def predict(self, X):
        pre_data = self.fit_transform(X)
        k = super().predict(pre_data)
        return k

    def fit(self, X, y):
        pre_data = self.fit_transform(X)
        k = super(PolynomialFeatures, self).fit(pre_data, y)
        return k


class SVRTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="SVM回归"):
        self.name = name
        self._model = SVR()

    def set_params(self, **new_params):
        """
        SVM回归
            重要参数：：kernel：核函数 str
		    epsilon: 正则化参数 float
		    全部参数：https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
        """
        super().set_params(**new_params)


class LinearRegressionTrain(ClassificationTrainFromSklearn):
    """
    包含Lasso & ridge
    """
    def __init__(self, name="线性回归"):
        self.name = name
        self._model = ElasticNet()

    def set_params(self, **new_params):
        """
        线性回归（包含Lasso & ridge）
        重要参数：alpha :正则化参数 float
        l1_ratio :lasso比例 取值范围[0,1] float
        max_iter : 最大迭代次数 int
        tol : 残差 float
        全部参数：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
        """
        super().set_params(**new_params)


class DecisionTreeRegressorTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="决策树回归"):
        self.name = name
        self._model = DecisionTreeRegressor()

    def set_params(self, **new_params):
        """
        决策树回归
        重要参数：criterion：优化目标函数，例如MSE， MAE等 str
        max_depth：最大深度 int or None
        min_samples_leaf ： 最小叶节点包含样本数 int（个数） or float（百分比）
        max_feature: 寻找最佳分割时要考虑的功能数量：int(个数） or float（百分比） or str(auto, sqrt, log2)
        全部参数： https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
        """
        super().set_params(**new_params)


class GradientBoostingRegressorTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="梯度提升树回归"):
        self.name = name
        self._model = GradientBoostingRegressor()

    def set_params(self, **new_params):
        """
         梯度提升树回归
        重要参数：
        loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, 损失函数 ls： least square， lad： least absolute deviation str
        learning_rate： 学习速率，迭代步长 float
        max_depth：最大深度 int
        n_estimators：树的个数 int
        max_depth：最大深度 int
        tol: 容差 float
        全部参数：https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn-ensemble-gradientboostingregressor
       """
        super().set_params(**new_params)


class MLPRegressorTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="BP神经网络回归"):
        self.name = name
        self._model = MLPRegressor()

    def set_params(self, **new_params):
        """
         BP神经网络回归
        重要参数：
        hidden_layer_sizes：每层神经元个数, length为隐藏层数-2  tuple
        activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’ 激活函数 str
        solver： {‘lbfgs’, ‘sgd’, ‘adam’}  优化方法 str
        alpha : L2正则化系数 float
        batch_size : 每次喂入神经网络样本数 int
        learning_rate : 迭代步长 float
        tol :残差 float
        max_iter ： 最大迭代步数 int
        API：https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
        """
        super().set_params(**new_params)


class IsotonicRegressionTrain(ClassificationTrainFromSklearn):
    """
    输入值只能包含一列
    """
    def __init__(self, name="保续回归"):
        self.name = name
        self._model = IsotonicRegression()

    def set_params(self, **new_params):
        """
        保续回归
        重要参数：
        increasing : 上升或下降 boolean
        API：https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression
        """
        super().set_params(**new_params)


class PolynomialRegressionTrain(ClassificationTrainFromSklearn):
    """
    多项式回归，此类为自制类，稳定性需测试
    """
    def __init__(self, name="多项式回归"):
        self.name = name
        self._model = PolynomialRegression()

    def set_params(self, **new_params):
        """
         多项式回归（不含Lasso & ridge）
         重要参数：degree : 多项式次数 int
         API：https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
        """
        super().set_params(**new_params)


class RandomForestRegressorTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="随机森林回归"):
        self.name = name
        self._model = RandomForestRegressor()

    def set_params(self, **new_params):
        """
         随机森林回归
        :param kwargs:参数键值
        重要参数：n_estimators :树的个数 int
        criterion :目标函数 str
        max_depth : 最大深度 int
        max_features : 特征选择方法 str
        max_leaf_nodes : 最大节点数 int
        API文档：
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        """
        super().set_params(**new_params)


if __name__=="__main__":
    jsn = {
            "model_params": {},
            "path": "D:/pro1/test2.csv",
            "data_columns": ["组织组", "总金额(元)", "销售组织单位", "运算加权总金额(元)", "客户类型", "客户级别",
                             "一级行业", "二级行业", "三级行业", "浪潮所属行业部", "行业部二级部门", "城市",
                             "预计招标类型", "项目公司1属性", "项目级别", "获得途径", "是否指名客户", "客户预算来源",
                             "客户预算", "商机产生背景", "采购特点", "技术方案是否已有", "是否存在竞争对手",
                             "评标环节可控制", "合作渠道招标方关系好", "用户或采购方表态支持", "标书指标倾向浪潮",
                             "总体规划方案已有", "项目招标方案已有", "是否老客户", "负责员工职位"],
            "label_columns": ["是否中标"],
            "save_path": "D:/pro1/model1.pkl"
           }

    import pandas as pd
    k = pd.read_csv("D:/pro1/test2.csv")
    test = LinearRegressionTrain()
    print(test.train_from_csv(**jsn))