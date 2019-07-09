#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import joblib
from sqlcon.sqlserver.sqlserver import SQLServer
from regression.regressionTrain import PolynomialRegression
from classification.Predict1 import ClassificationPredict
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression


class SVRPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, SVR):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class LinearRegressionPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, ElasticNet):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class DecisionTreeRegressorPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, DecisionTreeRegressor):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class GradientBoostingRegressorPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, GradientBoostingRegressor):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class MLPRegressorPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, MLPRegressor):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class IsotonicRegressionPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, IsotonicRegression):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class PolynomialRegressionPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, PolynomialRegression):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


class RandomForestRegressorPredict(ClassificationPredict):
    def load_model(self, **json):
        model_path = json['model_path']
        _model = joblib.load(model_path)
        if isinstance(_model, RandomForestRegressor):
            self._model = _model
        elif not hasattr(_model, "columns"):
            raise ValueError("所选模型不包含列信息")
        else:
            raise ValueError("类型不匹配")


if "__main__" == __name__:
    jsn = {
        "path": "D:/pro1/test2.csv",
        "model_path": "D:/pro1/model1.pkl",
        "save_path": "D:/pro1/response.csv"
    }
    test1 = RandomForestRegressorPredict()
    print(test1.predict_from_csv(**jsn))