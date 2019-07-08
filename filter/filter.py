#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sqlcon.sqlserver.sqlserver import SQLServer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import joblib

class filterBasic():
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
            self.fit_predict(features)
            self._model.columns = features.columns.values.tolist()
            self.save_model(json_file["save_path"]) #暂时保存
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_from_csv(self, **json):
        try:
            features = pd.read_csv(json["path"], usecols=json['data_columns'])
            self.set_params(**json["model_params"])
            self.fit(features)
            self._model.columns = json['data_columns']
            self.save_model(json["save_path"])  # 暂时保存
            print(self._model.offset_)
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_from_xls(self, **json):
        try:
            features = pd.read_excel(json["path"], usecols=json['data_columns'])
            self.set_params(**json["model_params"])
            self.fit(features)
            self._model.columns = json['data_columns']
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


class filterFromSklearn(filterBasic):
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

    def fit(self, x):
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return: 训练数据准确率
        """
        self._model.fit(x)


class IsolationForestFilter(filterFromSklearn):
    """
    "model_params": "n_estimators" : 包含树的个数，提高会导致训练时间增长，可能可以提升精度
                    "contamination": 可以反映过滤强度， 越大过滤强度越大
                    "max_samples": 每棵树包含的最大样本数
    """
    def __init__(self, name="孤立森林"):
        self._model = IsolationForest()
        self.name = name


class OneClassSVMFilter(filterFromSklearn):
    def __init__(self, name="SVM过滤"):
        self._model = IsolationForest()
        self.name = name


class Filter:
    def load_model(self, **json):
        model_path = json['model_path']
        self._model = joblib.load(model_path)

    def get_params(self, deep=True):
        """
        获得模型参数
        """
        return self._model.get_params(deep=deep)

    """
    def filter(self, data):
        columns = data.columns.values.tolist()
        for i in self._model.columns:
            if i not in columns:
                raise ValueError("输入值包含训练集中不存在的列")
        _data = data[self._model.columns]
        predata = pd.DataFrame(self._model.predict(_data))
        predata.columns = ["predata"]
        data = pd.concat([data, predata], axis=1)
        data = data[data["predata"] == 1]
        return data
    """

    def predict(self, data):
        columns = data.columns.values.tolist()
        for i in self._model.columns:
            if i not in columns:
                raise ValueError("输入值包含训练集中不存在的列")
        _data = data[self._model.columns]
        predata = pd.DataFrame(self._model.predict(_data))
        return predata

    def _connect_SQL(self, **json):
        """
        连接到SQL
        :param json_file: 入参
        :return:None
        """
        json_dict = json
        self._SQL = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'],
                              user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])

    def get_data_features(self, **json):
        """
        从数据库调取数据集
        :param json_file:入参， json
        :return: 仅含有特征变量的数据集 pd.dataFrame
        """
        json_dict = json
        data_features = self._SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'],
                                                    cols=json_dict['data_columns'])
        return data_features

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

    def preidct_from_xls(self, **json):
        try:
            self.load_model(**json)
            features = pd.read_excel(json["path"])
            pre = self.predict(features)
            """
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            """
            pre.columns = ["label"]
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_csv(self, **json):
        try:
            self.load_model(**json)
            features = pd.read_csv(json["path"])
            pre = self.predict(features)
            """
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            """
            pre.columns = ["label"]
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_sql(self, **json):
        try:
            self.load_model(**json)
            features = self.get_data_features(**json)
            pre = self.predict(features)
            """
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            """
            pre.columns = ["label"]
            pre.to_csv(json["save_path"], index=False)
            write = self.SQL.df_write_sqlserver(table=json['dbinfo']['outputtable'], df=pre,
                                                cols=json['data_columns'])
            pre.columns = ["label"]
            return {"info": write}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def get_label(self):
        pass


class IsolationForestFilterPredict(Filter):
    """
    "model_params": "n_estimators" : 包含树的个数，提高会导致训练时间增长，可能可以提升精度
                    "contamination": 可以反映过滤强度， 越大过滤强度越大
                    "max_samples": 每棵树包含的最大样本数
    """
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, IsolationForest):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class OneClassSVMFilterPredict(Filter):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, OneClassSVM):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class LocalOutlierFactorFilter:
    """
    训练与预测一体，没有单独的train和test接口
    关键参数：n_neighbors : int, optional (default=20)：参与预测的点的数量，无明显规律
    contamination": 可以反映过滤强度， 越大过滤强度越大
    """
    def __init__(self, name="局部异常因子"):
        self._model = LocalOutlierFactor()
        self.name = name

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

    def fit_predict(self, x):
        pass
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return: 训练数据准确率
        """
        return self._model.fit_predict(x)


    def _connect_SQL(self, **json_file):
        """
        连接到SQL
        :param json_file: 入参
        :return:None
        """
        json_dict = json_file
        self._SQL = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'],
                              user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])

    def get_data_label(self, **json_file):
        """
        从数据库调取数据集的标签
        :param json_file:
        :return: 仅含有标签的数据集 pd.dataFrame
        """
        json_dict = json_file
        data_label = self._SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'],
                                                 cols=json_dict['label_columns'])
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

    def train_predict_from_sql(self, **json_file):
        """
        训练模型并将模型保存
        :param json_file: 入参，json
        :return:是否成功
        """
        try:
            self._connect_SQL(**json_file)
            self.set_params(**json_file["model_params"])
            features = self.get_data_features(**json_file)
            pre = self.fit_predict(features)
            self._model.columns = features.columns.values.tolist()
            self.save_model(json_file["model_path"])  # 暂时保存
            pre.columns = ["label"]
            pre.to_csv(json_file["save_path"], index=False)
            write = self.SQL.df_write_sqlserver(table=json_file['dbinfo']['outputtable'], df=pre,
                                                cols=json_file['data_columns'])
            return {"info": write}
            return "success"
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_predict_from_csv(self, **json):
        try:
            features = pd.read_csv(json["path"], usecols=json['data_columns'])
            self.set_params(**json["model_params"])
            pre = pd.DataFrame(self.fit_predict(features))
            self._model.columns = json['data_columns']
            self.save_model(json["model_path"])  # 暂时保存
            pre.columns = ["label"]
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def train_predict_from_xls(self, **json):
        try:
            features = pd.read_excel(json["path"], usecols=json['data_columns'])
            self.set_params(**json["model_params"])
            pre = self.fit_predict(features)
            self._model.columns = json['data_columns']
            self.save_model(json["model_path"])  # 暂时保存
            pre.columns = ["label"]
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
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

    def load_model(self, **json):
        model_path = json['model_path']
        self._model = joblib.load(model_path)

"""
n = IsolationForest(contamination=0.001)
#contamination: 过滤强度， 越大过滤强度越大
iris_data = pd.read_csv("D:/pro1/Iris.csv")  # 数据路径
iris_train = iris_data.iloc[49:,:]
iris_test = iris_data.iloc[:48,:]
train = iris_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test = iris_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

print(n.fit_predict(train))
print(n.predict(test))
"""
if __name__ == "__main__":

    """
    jsn = {
        "model_params": {"contamination":0.05, "n_estimators":300,},
        "path": "D:/pro1/Iris1.csv",
        "data_columns": ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        "save_path": "model1.pkl"
    }

    x = IsolationForestFilter()
    x.train_from_csv(**jsn)
    jsn2 = {
        "model_path": "model1.pkl",
        "path": "D:/pro1/Iris2.csv",
        "save_path": "D:/pro1/result/IF0.05/filter.csv"
    }
    jsn3 = {
        "model_path": "model1.pkl",
        "path": "D:/pro1/Iris1.csv",
        "save_path": "D:/pro1/result/IF0.05/filter2.csv"
    }

    y = IsolationForestFilterPredict()
    print(y.predict_from_csv(**jsn2))
    print(y.predict_from_csv(**jsn3))

    """
    """
    LocalOutlierFactorFilter 参数：
    {
        "model_params": {"contamination":0.05},
        "data_columns": ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        "model_path": "model1.pkl",  /*模型保存路径*/
        "path": "D:/pro1/Iris1.csv", /*源数据路径*/
        "save_path": "D:/pro1/result/IF0.05/filter2.csv" /*预测数据保存路径*/
    }
    """


