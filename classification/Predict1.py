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
        for i in columns:
            if i not in self._model.columns:
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
            return {"info": "success", "pre_value" : pre}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_csv(self, **json):
        try:
            self.load_model(**json)
            features = pd.read_csv(json["path"], usecols=self._model.columns)
            pre = self._model.predict(features)
            return {"info": "success", "pre_value" : pre}
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def predict_from_sql(self, **json):
        try:
            self.load_model(**json)
            pre = self._model.predict(self.get_data_features(**json))
            return "success", pre
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
        "model_params": {},
        "path": "test2.csv",
        "data_columns": ['组织组', '总金额(元)', '销售组织单位', '运算加权总金额(元)', '客户类型', '客户级别',
                         '一级行业', '二级行业', '三级行业', '浪潮所属行业部', '行业部二级部门', '城市',
                         '预计招标类型', '项目公司1属性', '项目级别', '获得途径', '是否指名客户', '客户预算来源',
                         '客户预算', '商机产生背景', '采购特点', '技术方案是否已有', '是否存在竞争对手',
                         '评标环节可控制', '合作渠道招标方关系好', '用户或采购方表态支持', '标书指标倾向浪潮',
                         '总体规划方案已有', '项目招标方案已有', '是否老客户', '负责员工职位'],
        "label_columns": ['是否中标'],
        "model_path": "model.pkl"
    }
    test1 = LogisticRegressionPredict()
    print(test1.predict_from_csv(**jsn))


