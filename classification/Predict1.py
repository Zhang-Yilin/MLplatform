#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import joblib
from sqlcon.sqlserver.sqlserver import SQLServer
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


class ClassificationPredict:

    def load_model(self, **json):
        model_path = json['model_path']
        self._model = joblib.load(model_path)

    def get_params(self, deep=True):
        """
        获得模型参数
        """
        return self._model.get_params(deep=deep)

    def predict(self, data):
        columns = data.columns.values.tolist()
        for i in self._model.columns:
            if i not in columns:
                raise ValueError("输入值包含训练集中不存在的列")
        _data = data[self._model.columns]
        return self._model.predict(_data)

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

    def get_data_label(self, **json):
        """
        从数据库调取数据集的标签
        :param json_file:
        :return: 仅含有标签的数据集 pd.dataFrame
        """
        json_dict = json
        data_label = self._SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'],cols=json_dict['label_columns'])
        if data_label.shape[1] != 1:
            raise ValueError("错误：标签列数不为1")
        return data_label

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
            features = pd.read_excel(json["path"], usecols=self._model.columns)
            pre = self._model.predict(features)
            predf = pd.DataFrame(pre)
            predf.columns = self._model.label
            predf.to_csv(json["save_path"], index=False)
            return {"info": "success", "pre_value" : pre}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_csv(self, **json):
        try:
            self.load_model(**json)
            features = pd.read_csv(json["path"], usecols=self._model.columns)
            pre = self._model.predict(features)
            predf = pd.DataFrame(pre)
            predf.columns = self._model.label
            predf.to_csv(json["save_path"], index=False)
            return {"info": "success", "pre_value" : pre}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_sql(self, **json):
        try:
            self.load_model(**json)
            pre = self._model.predict(self.get_data_features(**json))
            predf = pd.DataFrame(pre)
            predf.columns = self._model.label
            predf.to_csv(json["save_path"], index=False)
            write = self.SQL.df_write_sqlserver(table=json['dbinfo']['outputtable'], df=pre,
                                                cols=json['data_columns'])
            return {"info": write, "pre_value" : pre}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)


class DecisionTreePredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, DecisionTreeClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class LogisticRegressionPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, LogisticRegression):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class RandomForestPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, RandomForestClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class GradientBoostingPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, GradientBoostingClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class MLPPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, MLPClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class LinearDiscriminantAnalysisPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, LinearDiscriminantAnalysis):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class SVMPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, SVC):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class KNNPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, KNeighborsClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class XGradientBoostingPredict(ClassificationPredict):
    def load_model(self,**json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, XGBClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")

class AdaBoostPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, AdaBoostClassifier):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")




if __name__ == "__main__":
    jsn = {
        "path": "D:/pro1/test2.csv",
        "model_path": "D:/pro1/model1.pkl",
        "save_path": "D:/pro1/response.csv"
    }
    test1 = LogisticRegressionPredict()
    print(test1.predict_from_csv(**jsn))


