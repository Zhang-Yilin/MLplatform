#!/usr/bin/env python
# -*- coding:utf-8 -*-

import joblib
import warnings
import copy
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sqlcon.sqlserver.sqlserver import SQLServer
from sklearn import tree


class ClassificationTrain:

    def __get__(self):
        return copy.deepcopy(self)

    def get_params(self, deep=True):
        pass
    """
    def _get_valid_params(self):
        pass
    """

    def set_params(self, **new_params):
        pass

    def fit(self, x, y):
        pass

    def self_predict(self, x):
        pass

    def _connect_SQL(self, **json_file):
        """
        连接到SQL
        :param json_file: 入参
        :return:None
        """
        json_dict = json_file
        self._SQL = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'], user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])

    def get_data_label(self, **json_file):
        """
        从数据库调取数据集的标签
        :param json_file:
        :return: 仅含有标签的数据集 pd.dataFrame
        """
        json_dict = json_file
        data_label = self._SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'],cols=json_dict['label_columns'])
        if data_label.shape[1] != 1:
            raise ValueError("错误：标签列数不为1")
        return data_label

    def get_data_features(self, **json_file):
        """
        从数据库调取数据集
        :param json_file:入参， json
        :return: 仅含有特征变量的数据集 pd.dataFrame
        """
        json_dict = json_file
        data_features = self._SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'],
                                                    cols=json_dict['data_columns'])
        return data_features

    def train_from_sql(self, **json_file):
        """
        训练模型并将模型保存
        :param json_file: 入参，json
        :return:是否成功
        """
        try:
            self._connect_SQL(**json_file)
            self.set_params(**json_file["model_params"])
            features = self.get_data_features(**json_file)
            label = self.get_data_label(**json_file).values.ravel()
            self.fit(features, label)
            pre_data = (self.self_predict(features))
            """
            pre_data为预测信息
            可输出到表
            """
            self._model.columns = features.columns.values.tolist()
            self._model.label = label.columns.values.tolist()
            self.save_model(json_file["save_path"]) #暂时保存
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_from_csv(self, **json):
        try:
            features = pd.read_csv(json["path"], usecols=json['data_columns'])
            labels = pd.read_csv(json["path"], usecols=json['label_columns']).values.ravel()
            self.set_params(**json["model_params"])
            self.fit(features, labels)
            pre_data = (self.self_predict(features))
            """
            pre_data为预测信息
            可输出到表
            """
            self._model.columns = json['data_columns']
            self._model.label = json["label_columns"]
            self.save_model(json["save_path"])  # 暂时保存
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_from_xls(self, **json):
        try:
            features = pd.read_excel(json["path"], usecols=json['data_columns'])
            labels = pd.read_excel(json["path"], usecols=json['label_columns']).values.ravel()
            self.set_params(**json["model_params"])
            self.fit(features, labels)
            pre_data = (self.self_predict(features))
            """
            pre_data为预测信息
            可输出到表
            """
            self._model.columns = json['data_columns']
            self._model.label = json["label_columns"]
            self.save_model(json["save_path"])  # 暂时保存
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def save_model(self, model_path):
        """
        保存模型
        :param model_path: 模型保存路径
        :return:是否成功
        """
        try:
            joblib.dump(self._model, model_path)
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def get_model(self):
        """
        调用模型
        :return:模型
        """
        try:
            return self._model
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)


class ClassificationTrainFromSklearn(ClassificationTrain):
    def get_params(self, deep=True):
        """
        获得模型参数
        """
        return self._model.get_params(deep=deep)

    def _get_valid_params(self):
        """
        获取有效参数
        :return: List
        """
        param = self.get_params()
        return [i for i in param.keys()]

    def set_params(self, **new_params):
        """
        设置模型参数
        :param new_params: 模型参数键值
        只将模型参数包含的超参赋值给模型
        :return:
        """
        for k in new_params.keys():
            if k not in self._get_valid_params():
                raise ValueError("传入参数含有模型中不包含的参数")
                break
        feed_dict = {k: v for k, v in new_params.items() if k in self._get_valid_params()}
        if len(feed_dict) == 0:
            warnings.warn("模型参数未被修改")
        self._model.set_params(**feed_dict)

    def fit(self, x, y):
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return:
        """
        self._model.fit(x, y)

    def self_predict(self, x):
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return: 训练数据预测标签
        """
        return self._model.predict(x)


class DecisionTreeTrain(ClassificationTrainFromSklearn):

    def __init__(self, name="决策树"):
        self.name = name
        self._model = DecisionTreeClassifier()

    def set_params(self, **new_params):
        """
        可选参数：criterion : 分类标准
                max_depth : 最大深度

        :return:
        """
        super().set_params(**new_params)

    def get_importantce(self):
        return self._model.feature_importances_

    def get_tree(self):
        return self._model.tree_

    def get_plot(self):
        pass


class LogisticRegressionTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="逻辑回归"):
        self.name = name
        self._model = LogisticRegression()
    def set_params(self, **new_params):
        """
        可选参数：penalty：惩罚函数类型（字符型，'l1'或'l2'，默认为'l2'）
                max_iter：最大迭代次数（整数型，默认100）
                C：正则化参数（值越小正则力度越大，浮点型，默认1））
                tol：收敛容差（浮点型，默认1e-4）
        :return:
        """
        super().set_params(**new_params)

    def return_importance(self, type = "positive", ):
        pass


class RandomForestTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="随机森林"):
        self.name = name
        self._model = RandomForestClassifier()
    def set_params(self, **new_params):
        """
        可选参数：n_estimators：决策树个数（整数型，默认10）
                max_depth：最大深度（整数型，默认None）
                criterion：信息度量方式（字符型，‘gini’或'entropy'，默认'gini'）
                max_features：特征选择方法（整数，浮点或字符型，‘auto’或'sqrt'或'log2'或None，默认'auto'）
                random_state：随机数生成器（整数型，默认0）
        :return:
        """
        super().set_params(**new_params)

    def get_importantce(self):
        return self._model.feature_importances_


class GradientBoostingTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="梯度提升决策树"):
        self.name = name
        self._model = GradientBoostingClassifier()
    def set_params(self, **new_params):
        """
        可选参数：n_estimators：决策树个数（整数型，默认100）
                max_depth：最大深度（整数型，默认3）
                learning_rate：学习速率（浮点型，默认 0.1）
                random_state：随机数生成器（整数型，默认None）
        :return:
        """
        super().set_params(**new_params)

    def get_importantce(self):
        return self._model.feature_importances_


class MLPTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="神经网络"):
        self.name = name
        self._model = MLPClassifier()
    def set_params(self, **new_params):
        """
        可选参数：Activation：激活函数（字符型，'identity', 'logistic', 'tanh', 'relu'，默认'relu'）
                learning_rate_init：学习速率（浮点型，默认0.001）
                hidden_layer_sizes：隐层大小（tuple，默认(100,)）
        :return:
        """
        super().set_params(**new_params)


class LinearDiscriminantAnalysisTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="线性判别"):
        self.name = name
        self._model = LinearDiscriminantAnalysis()
    def set_params(self, **new_params):
        """
        可选参数：使用默认参数即可
        :return:
        """
        super().set_params(**new_params)


class SVMTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="SVM"):
        self.name = name
        self._model = SVC()

    def set_params(self, **new_params):
        """
        可选参数：C：惩罚系数（浮点型，默认1.0）
                Kernel：核函数（字符型，'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable，默认‘rbf’）
        :return:
        """
        super().set_params(**new_params)


class KNNTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="KNN"):
        self.name = name
        self._model = KNeighborsClassifier()

    def set_params(self, **new_params):
        """
        可选参数：n_neighbors：最近邻个数（整数型，默认5）
                p：距离度量方式（整数型，1 = 曼哈顿距离，2=欧几里得距离，默认2）
        :return:
        """
        super().set_params(**new_params)


class XGradientBoostingTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="Xgboost"):
        self.name = name
        self._model = XGBClassifier()

    def set_params(self, **new_params):
        """
        可选参数：Max_depth：最大深度（整数型，默认3）
                Learning_rate：学习速率（浮点型，默认 0.1）
                n_estimators：最大迭代次数（整数型，默认100）
                random_state：随机数生成器（整数型，默认0）
                reg_alpha：L1正则化系数（浮点型，默认0）
                reg_lambda：L2正则化系数（浮点型，默认1）
        :return:
        """
        super().set_params(**new_params)

        def get_importantce(self):
            return self._model.feature_importances_


class AdaBoostTrain(ClassificationTrainFromSklearn):
    def __init__(self, name="Adaboost"):
        self.name = name
        self._model = AdaBoostClassifier()

    def set_params(self, **new_params):
        """
        可选参数： n_estimators：分类器个数（整数型，默认50）
                 learning_rate：学习效率（浮点型，默认1）
        :return:
        """
        super().set_params(**new_params)



if __name__ == "__main__":
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


    test = LogisticRegressionTrain()
    test.train_from_csv(**jsn)
    t = test.get_model()

    import sys
    import csv
    import os

    input_path = 'D:/pro1/Iris.csv'  # campaign 基础文件固定位置
    with open(input_path, 'a') as input_csv:
        x = pd.read_csv('D:/pro1/Iris.csv')
        v = list(x.iloc[:,3])
        csv_write = csv.writer(input_csv)
        csv_write.writerow(v)




