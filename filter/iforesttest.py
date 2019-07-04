#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sqlcon.sqlserver.sqlserver import SQLServer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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
    def __init__(self, name="孤立森林"):
        self._model = IsolationForest()
        self.name = name
    # contamination: 过滤强度， 越大过滤强度越大


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
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_csv(self, **json):
        try:
            self.load_model(**json)
            features = pd.read_csv(json["path"])
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            pre.to_csv(json["save_path"], index=False)
            return {"info": "success"}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_sql(self, **json):
        try:
            self.load_model(**json)
            features = self.get_data_features(**json)
            if json["output"] == 0:
                pre = self.filter(features)
            elif json["output"] == 1:
                pre = self.predict(features)
            else:
                raise ValueError("选择的输出方式不存在")
            pre.to_csv(json["save_path"], index=False)
            write = self.SQL.df_write_sqlserver(table=json['dbinfo']['outputtable'], df=pre,
                                                cols=json['data_columns'])
            return {"info": write}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def get_label(self):
        pass


class IsolationForestFilterPredict(Filter):
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
    jsn = {
        "model_params": {"contamination":0.4},
        "path": "D:/pro1/Iris.csv",
        "data_columns": ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        "save_path": "model1.pkl"
    }

    x = IsolationForestFilter()
    x.train_from_csv(**jsn)
    jsn2 = {
        "model_path": "model1.pkl",
        "path": "D:/pro1/Iris.csv",
        "save_path": "filter.csv",
        "output": 0
    }
    iris_data = pd.read_csv("D:/pro1/Iris.csv")  # 数据路径
    iris_train = iris_data.iloc[49:,:]
    iris_test = iris_data.iloc[:148,:]
    train = iris_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    test = iris_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    y = IsolationForestFilterPredict()
    print(y.predict_from_csv(**jsn2))