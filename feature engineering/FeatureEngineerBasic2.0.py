#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author：Zhang Yilin
"""
包含主成分分解，特征值分解，因子分析，分箱
暂时不能处理缺失值
"""

import abc
import copy
import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_extension_array_dtype,is_categorical_dtype
from factor_analyzer import FactorAnalyzer
import pandas.core.reshape.tile
from pandas._libs.lib import is_integer
from pandas.core.algorithms import quantile
from pandas.core.algorithms import take_nd, unique, ensure_int64
from pandas.core.arrays.categorical import Categorical
from sqlserver import SQLServer


class Feature(object):
    """
    特征工程基本操作
    """
    @abc.abstractmethod
    def set_data(self, data_in):
        """
        读取数据
        """
        pass

    def get_numeric_data(self):
        """
        返回输入类中的数值变量列名
        :param self:
        :return: 数值类列名 list
        """
        _dtype = list(self.data.dtypes)
        _name = list(self.data.columns.values.tolist())
        response = list()
        for i in range(len(_dtype)):
            if "int" in str(_dtype[i]) or "float" in str(_dtype[i]) or "complex" in str(_dtype[i]):
                response.append(_name[i])
        return response

    def get_cate_data(self):
        """
        返回非数值类变量列名
        :return: 非数值类变量列名 list
        """
        _dtype = list(self.data.dtypes)
        _name = list(self.data.columns.values.tolist())
        response = list()
        for i in range(len(_dtype)):
            if "int" not in str(_dtype[i]) and "float" not in str(_dtype[i]) and "complex" not in str(_dtype[i]):
                response.append(_name[i])
        return response

    def _check_missing_value(self):
        """
        返回不含缺失值的列名
        :return: 不含缺失值列的列名 list
        """
        list_full = list(self.data.isnull().any())
        response = list()
        for i in range(len(list_full)):
            if list_full[i] == False:
                response.append(self.data.columns.values.tolist()[i])
        return response

    def _check_input(self):
        """
        检查输入值是否为空
        :return: bool
        """
        return not(self.data == None)

    def _connect_SQL(self, **json_file):
        """
        连接到sql
        :param json_file:传入参数
        :return:
        """
        json_dict = json_file
        self.SQL = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'], user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])



    def get_data_features(self, **json_file):
        """
        从数据库调取数据集
        :param json_file:
        :return: 仅含有特征变量的数据集
        """
        json_dict = json_file
        data = self.SQL.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'])
        return data

    @abc.abstractmethod
    def _select_by(self, **json_file):
        """
        设定特征工程选择输出的方式，如主成分分析中按个数输出或者按解释方差的百分比输出
        :param type_arg:
        :return:
        """
        pass

    def transform_from_SQL(self, **json_file):
        """
        从sql获取数据，并按传入参数变换数据，并将结果按传入指令写入sql
        :param json_file:
        :return:
        """
        try:
            json_dict = json_file
            self._connect_SQL(**json_file)
            self.set_data(self.get_data_features(**json_file))
            feature_data = self._select_by(**json_file)
            #write = self.SQL.df_write_sqlserver(table=json_dict['dbinfo']['outputtable'], df=feature_data,
            #                                   cols=feature_data.columns.values.tolist())
            return feature_data
        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)

    def select_column(self, *colname):
        """
        保存选中列
        :param colname:所有输入的列名
        维护不含缺失值的数值类的列名
        """
        for i in colname:
            if i not in self.data.columns.values.tolist():
                raise ValueError("所选列不存在")
            elif i not in self.full_data:
                raise ValueError("所选列存在缺失值")
            elif i not in self.numeric_data:
                raise ValueError("所选列不为数值")
            elif i in self.selected_column:
                raise ValueError("this column has been selected")
            else:
                self.selected_column.append(i)

    def remove_column(self, *colname):
        """
        从选中列中移除列
        :param colname: 需要移除的列
        维护移除后的所选择列的列名
        """
        for i in colname:
            if i not in self.selected_column:
                raise ValueError("该列未被选中")
            else:
                self.selected_column.remove(i)

    def reset_column(self):
        """
        清空所选列
        """
        self.selected_column=[]

    def get_model(self):
        """
        返回模型
        :return: 模型
        """
        return self.model


class PCA_select(Feature):
    """
    主成分分析
    """
    data = None
    selected_column = list()

    def set_data(self, **json_file):
        """
        传入数据
        :param data: 需要处理的数据
        :return:
        """
        data = json_file['PCA']['data']
        self.data = data
        self.full_data = self._check_missing_value()
        self.numeric_data = self.get_numeric_data()

    def fit(self):
        """
        对传入数据进行主成分分析，并得到所有的主成分变量
        :return: 变换完成的所有主成分向量 pandas.dataFrame
        """
        feed_data = self.data[self.selected_column]
        self.model = PCA(n_components = feed_data.shape[1])
        self.model.fit(feed_data)
        return self.model.transform(feed_data)

    def select_by_number(self, **json_file):
        """
        选择特征值最大的num个向量
        :param num: 选择向量个数 int 大于0，小于等于输入变量个数
        :return: 所有输入列和选定的PCA向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        num = json_file['PCA']['num']
        if num < 0 or num > len(self.selected_column):
            raise ValueError("too many or too less columns are selected")
        temp = self.fit()
        result = pd.DataFrame(temp[:, 0:num])
        colnames = list()
        for i in range(num):
            colnames.append("PCA "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """
        remained = [i for i in self.data.columns.values.tolist() if i not in self.selected_column]
        for i in remained:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """

    def select_by_percentage(self, **json_file):
        """
        按向量解释方差的百分比选择
        :param percentage: 解释的百分比 [0,1] float
        :return: 所有输入列和选定的PCA向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        percentage = json_file['PCA']['percentage']
        if percentage < 0 or percentage > 1:
            raise ValueError("percentage must be in [0,1]")
        temp = self.fit()
        per_list = self.model.explained_variance_ratio_
        index = 0
        sum = 0
        while sum < percentage:
            sum += per_list[index]
            index += 1
        result = pd.DataFrame(temp[:,0:index])
        colnames = list()
        for i in range(index):
            colnames.append("PCA "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """
        remained = [i for i in self.data.columns.values.tolist() if i not in self.selected_column]
        for i in remained:
            result.insert(0, column=i, value=self.data[[i]])
        return result

        """

    def _select_by(self, **json_file):
        """
        按输入参数返回PCA结果
        :param type_arg: 控制变量字典
        字典中"type" == 0: 按数量选择， typearg: 特征向量的个数 int 大于0， 小于输入变量个数
        字典中"type" == 1: 按解释方差的比例选择， typearg: 解释方差的比例 float 大于等于0， 小于等于1
        :return: 所有输入列和选定的PCA向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """

        if json_file['PCA']["type"] == 0:
            self.select_column(*json_file['PCA']["columns"])
            return self.select_by_number(**json_file)
        elif json_file['PCA']["type"] == 1:
            self.select_column(*json_file['PCA']["columns"])
            return self.select_by_percentage(**json_file)
        else:
            raise ValueError("type error：不存在所选类")


class SVD_select(Feature):
    """
    奇异值分解
    """
    data = None
    selected_column = list()

    def set_data(self, **json_file):
        """
        传入数据
        :param data: 需要处理的数据
        :return:
        """
        data = json_file['SVD']['data']
        self.data = data
        self.full_data = self._check_missing_value()
        self.numeric_data = self.get_numeric_data()

    def fit(self):
        """
        对传入数据进行奇异值，并得到所有的奇异值变量
        :return: 变换完成的所有奇异值向量 pandas.dataFrame
        """
        feed_data = self.data[self.selected_column]
        self.model = TruncatedSVD(feed_data.shape[1]-1)
        self.model.fit_transform(feed_data)
        return self.model.transform(feed_data)

    def select_by_number(self, **json_file):
        """
        选择奇异值最大的num个向量
        :param num: 选择向量个数 int 大于0，小于输入变量个数
        :return: 所有输入列和选定的奇异值向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        num = json_file['SVD']['num']
        if num < 0 or num > len(self.selected_column):
            raise ValueError("too many or too less columns are selected")
        temp = self.fit()
        result = pd.DataFrame(temp[:,0:num])
        colnames = list()
        for i in range(num):
            colnames.append("SVD "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """
        remained = [i for i in self.data.columns.values.tolist() if i not in self.selected_column]
        for i in remained:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """

    def select_by_percentage(self, **json_file):
        """
        按向量解释方差的百分比选择
        :param percentage: 解释的百分比 [0,1] float
        :return: 所有输入列和选定的奇异值向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        percentage = json_file['SVD']['percentage']
        if percentage < 0 or percentage > 1:
            raise ValueError("percentage must be in [0,1]")
        temp = self.fit()
        per_list = self.model.explained_variance_ratio_
        index = 0
        sum = 0
        while sum < percentage:
            sum += per_list[index]
            index += 1
        result = pd.DataFrame(temp[:,0:index])
        colnames = list()
        for i in range(index):
            colnames.append("SVD "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result
        """
        remained = [i for i in self.data.columns.values.tolist() if i not in self.selected_column]
        for i in remained:
            result.insert(0, column=i, value=self.data[[i]])
        return result

        """
    """
    def list_all_member(self):
        for name,value in vars(self).items():
            print('%s=%s'%(name,value))
    """
    def _select_by(self, **json_file):
        """
        按输入参数返回SVD结果
        :param type_arg: 控制变量字典
        字典中"type" == 0: 按数量选择， typearg: 特征向量的个数 int 大于0， 小于输入变量个数
        字典中"type" == 1: 按解释方差的比例选择， typearg: 解释方差的比例 float 大于等于0， 小于等于1
        :return: 所有输入列和选定的SVD向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """

        if json_file['SVD']["type"] == 0:
            self.select_column(*json_file['SVD']["columns"])
            return self.select_by_number(**json_file)
        elif json_file['SVD']["type"] == 1:
            self.select_column(*json_file['SVD']["columns"])
            return self.select_by_percentage(**json_file)
        else:
            raise ValueError("type error：不存在所选类")


class Factor_Analyse_select(Feature):
    """
    因子分析
    """
    data = None
    selected_column = list()
    method = 'minres'
    def set_data(self, **json_file):
        """
        传入数据
        :param data: 需要处理的数据
        :return:
        """
        data = json_file['FA']['data']
        self.data = data
        self.full_data = self._check_missing_value()
        self.numeric_data = self.get_numeric_data()
    """
    def __init__(self, data):
        self.set_data(data)
    
    def select_column(self, *colnames):
        for colname in colnames:
            if colname not in self.data.columns.values.tolist():
                raise ValueError("所选列不存在")
            elif colname not in self.full_data:
                raise ValueError("所选列存在缺失值")
            elif colname not in self.numeric_data:
                raise ValueError("所选列不为数值")
            elif colname in self.selected_column:
                raise ValueError("this column has been selected")
            else:
                self.selected_column.append(colname)
    """
    def set_method(self, **json_file):
        """
        设置方法
        :param method: 选择的方法 str 可选："minres", "ml", "principal"
        :return:
        """
        method = json_file['FA']['method']
        if method is None:
            warnings.warn("参数未设置")
        elif method not in ['minres', 'ml', 'principal']:
            raise ValueError("invalid method")
        else:
            self.method = method

    def fit(self):
        """
        按所选方法进行变换
        :return: 变换完毕的所有向量
        """
        feed_data = self.data[self.selected_column]
        self.model = FactorAnalyzer(n_factors=feed_data.shape[1], method=self.method, rotation=None)
        self.model.fit(feed_data)
        return self.model.transform(feed_data)

    def select_by_number(self, **json_file):
        """
        选择特征值最大的num个向量
        :param num: 选择向量个数 int
        :return: 所有输入列和选定的因子向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        num = json_file['FA']['num']
        if num < 0 or num > len(self.selected_column):
            raise ValueError("too many or too less columns are selected")
        temp = self.fit()
        result = pd.DataFrame(temp[:,0:num])
        colnames = list()
        for i in range(num):
            colnames.append("FA "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result

    def select_by_eig_GE_1(self):
        """
        选择特征值大于1的因子
        :return: 所有输入列和选定的因子向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        pre_list = self.model.get_eigenvalues()
        index = 0
        for i in pre_list[0]:
            if i < 1:
                break
            index += 1
        temp = self.fit()
        result = pd.DataFrame(temp[:,0:index])
        colnames = list()
        for i in range(index):
            colnames.append("FA "+str(i+1))
        result.columns = colnames
        for i in self.data.columns.values.tolist()[::-1]:
            result.insert(0, column=i, value=self.data[[i]])
        return result

    def _select_by(self, **type_arg):
        """
        按输入参数返回因子分析结果
        :param type_arg: 控制变量字典
        字典中"method": 因子分析的方法 "minres":最小残差法(默认), "ml":极大似然, "principal"：主成分分析
        字典中"type" == 0: 按数量选择结果， typearg: 选择特征值最大的typearg个因子
        字典中"type" == 1: 选择特征值大于1的所有向量
        :return: 所有输入列和分箱结果向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        if "method" in json['FA']['type_arg'].keys():
            self.set_method(**json['FA']['type_arg']["method"])
        if json['FA']['type_arg']["type"] == 0:
            self.select_column(*json['FA']['type_arg']["columns"])
            return self.select_by_number(**json['FA']['type_arg']["typearg"])
        elif json['FA']['type_arg']["type"] == 1:
            self.select_column(*json['FA']['type_arg']["columns"])
            return self.select_by_eig_GE_1()
        else:
            raise ValueError("type error：不存在所选类")


class cut_select(Feature):
    """
    分箱
    """
    data = None
    selected_column = list()
    method = 'width'
    suffix = "_bin"

    def set_data(self, **json):
        """
        传入数据
        :param data: 需要处理的数据
        :return:
        """
        data = json['Cut_select']['data']
        self.data = data
        self.full_data = self._check_missing_value()
        self.numeric_data = self.get_numeric_data()
    """
    def __init__(self, data):
        self.set_data(data)
    """
    """
    def select_column(self, *colnames):
        for colname in colnames:
            if colname not in self.data.columns.values.tolist():
                raise ValueError("所选列不存在")
            elif colname not in self.full_data:
                raise ValueError("所选列存在缺失值")
            elif colname not in self.numeric_data:
                raise ValueError("所选列不为数值")
            elif colname in self.selected_column:
                raise ValueError("this column has been selected")
            else:
                self.selected_column.append(colname)
    """
    '''
    def set_method(self, method):
        """
        设置分箱方法
        :param method: 方法 str "width"按设定宽度分箱, "number"按设定分箱数分箱, "quantile"按分位点分箱, "sd"按标准差分箱
        :return: 
        """
        if method not in ['width', 'number', 'quantile', 'sd']:
            raise ValueError("invalid method")
        else:
            self.method = method
    '''

    def set_suffix(self, **json):

        """
        设置输出变量的后缀
        :param strs: 后缀 str
        :return:
        """
        strs = json['Cut_select']['strs']
        self.suffix = str(strs)

    def _cut_width(self, **json):
        """
        按设定宽度分箱
        :param width:分箱宽度，与列名对应 list[int]
        :return:包含分箱结果与输入表的数据框 pandas.dataFrame
        """
        width = json['Cut_select']['width']
        iter1 = self.data[self.selected_column].__iter__()
        result = self.data
        cut_dict = dict(zip(iter1, *width.__iter__()))
        for i, w in cut_dict.items():
            _min = min(self.data[i])
            _max = max(self.data[i])
            j = 0
            bin_list = []
            while _min + j*w < _max:
                bin_list.append((_min+j*w, _min+(j+1)*w))
                j += 1
            bins = pd.IntervalIndex.from_tuples(bin_list, closed="left")
            result.insert(0, column=i+self.suffix, value=pd.cut(self.data[i], bins, include_lowest=True))
        del cut_dict
        return result

    def _cut_number(self, **json):
        """
        按设定分箱数分箱
        :param num:分箱数，与列名对应 list[int(正整数）]
        :return:包含分箱结果与输入表的数据框 pandas.dataFrame
        """
        num = json['Cut_select']['num']
        iter1 = self.data[self.selected_column].__iter__()
        result = self.data
        cut_dict = dict(zip(iter1, *num.__iter__()))
        for i, n in cut_dict.items():
            #temp = pd.cut(self.data[i], num, include_lowest=True, duplicates="drop")
            result.insert(0, column=i + self.suffix,
                          value=my_cut(self.data[i], n, include_lowest=True, right=False))
            #del temp
        del cut_dict
        return result

    def _cut_quantile(self, **json):
        """
        按设定分位数分箱
        :param quantile:分位数，与列名对应 list[int(正整数）] eg：quantile = 3；分箱为 0-33% 33%-67% 67%-100%
        :return:包含分箱结果与输入表的数据框 pandas.dataFrame
        """
        quantile = json['Cut_select']['quantile']
        iter1 = self.data[self.selected_column].__iter__()
        result = self.data
        cut_dict = dict(zip(iter1, *quantile.__iter__()))
        for i, q in cut_dict.items():
            result.insert(0, column=i + self.suffix, value=my_qcut(self.data[i], q, duplicates="drop"))
        del cut_dict
        return result

    def _cut_sd(self, **json):
        """
        按标准差分箱
        :param nsd: 标准差倍数 ，与列名对应 list[int(正整数）]
        :return: 包含分箱结果与输入表的数据框 pandas.dataFrame
        分箱节点保留三位小数
        """
        nsd = json['Cut_select']['nsd']
        iter1 = self.data[self.selected_column].__iter__()
        result = self.data
        cut_dict = dict(zip(iter1, *nsd.__iter__()))
        for i, n in cut_dict.items():
            _mean = np.mean(self.data[i])
            _sd = np.std(self.data[i])
            bin_cut = [-float("inf"), float("inf")]
            bin_list = []
            for j in range(n):
                bin_cut.append(round(_mean + (j+1)*_sd, 3))
                bin_cut.append(round(_mean - (j+1)*_sd, 3))
            bin_cut.sort()
            for k in range(len(bin_cut)-1):
                bin_list.append((bin_cut[k], bin_cut[k+1]))
            bins = pd.IntervalIndex.from_tuples(sorted(bin_list), closed="left")
            result.insert(0, column=i + self.suffix, value=pd.cut(self.data[i], bins, include_lowest=True))
        return result

    def _select_by(self, **json):
        """
        按输入参数返回分箱结果
        :param json['Cut_select']['type_arg']: 控制变量字典
        字典中"suffix": 分箱结果列名的后缀，默认为"_bin"
        字典中"type" == 0: 按数量分箱， typearg: 分箱数，与列名对应 list[int(正整数）]
        字典中"type" == 1: 按宽度分箱， typearg: 分箱宽度，与列名对应 list[int]
        字典中"type" == 2: 按分位数分箱， typearg: 分位数，与列名对应 list[int(正整数）] eg：typearg = 3；分箱为 0-33% 33%-67% 67%-100%
        字典中"type" == 3: 按标准差分箱， typearg: 标准差倍数，与列名对应 list[int]
            eg: typearg = 1: 分箱为 -inf-（均值-一倍标准差），（均值-一倍标准差）-（均值+一倍标准差），（均值-一倍标准差）-inf
        :return: 所有输入列和分箱结果向量组成的数据框（包含输入表的所有数据） pandas.dataFrame
        """
        if len(json['Cut_select']['type_arg']["columns"]) != len(json['Cut_select']['type_arg']["typearg"]):
            raise ValueError("列数与分箱参数不匹配")
        if "suffix" in json['Cut_select']['type_arg'].keys():
            self.set_suffix(**json['Cut_select']['type_arg']["suffix"])
        if json['Cut_select']['type_arg']["type"] == 0:
            self.select_column(json['Cut_select']['type_arg']["columns"])
            return self._cut_number(**json['Cut_select']['type_arg']["typearg"])
        elif json['Cut_select']['type_arg']["type"] == 1:
            self.select_column(json['Cut_select']['type_arg']["columns"])
            return self._cut_width(**json['Cut_select']['type_arg']["typearg"])
        elif json['Cut_select']['type_arg']["type"] == 2:
            self.select_column(json['Cut_select']['type_arg']["columns"])
            return self._cut_quantile(**json['Cut_select']['type_arg']["typearg"])
        elif json['Cut_select']['type_arg']["type"] == 3:
            self.select_column(json['Cut_select']['type_arg']["columns"])
            return self._cut_sd(**json['Cut_select']['type_arg']["typearg"])
        else:
            raise ValueError("type error：不存在所选类")

"""
以下三个函数为重写的函数，使用这些函数是为了满足分箱结果的格式要求
"""
def my_cut(x, bins, right=True, labels=None, retbins=False, precision=3,
        include_lowest=False, duplicates='raise'):

    x_is_series, series_index, name, x = pandas.core.reshape.tile._preprocess_for_cut(x)
    x, dtype = pandas.core.reshape.tile._coerce_to_type(x)

    if not np.iterable(bins):
        if pd._libs.lib.is_scalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")

        try:  # for array-like
            sz = x.size
        except AttributeError:
            x = np.asarray(x)
            sz = x.size

        if sz == 0:
            raise ValueError('Cannot cut empty array')
        nanmin = pd.core.nanops._nanminmax('min', fill_value_typ='+inf')
        nanmax =  pd.core.nanops._nanminmax('max', fill_value_typ='-inf')
        rng = (nanmin(x), nanmax(x))
        mn, mx = [mi + 0.0 for mi in rng]


        if mn == mx:  # adjust end points before binning
            mn -= .001 * abs(mn) if mn != 0 else .001
            mx += .001 * abs(mx) if mx != 0 else .001
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
        else:  # adjust end points after binning
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
            bins2 = copy.deepcopy(bins)
            adj = (mx - mn) * 0.001  # 0.1% of the range
            if right:
                bins[0] -= adj
            else:
                bins[-1] += adj
    elif isinstance(bins, pd.IntervalIndex):
        if bins.is_overlapping:
            raise ValueError('Overlapping IntervalIndex is not accepted.')

    else:
        if pd.core.dtypes.common.is_datetime64tz_dtype(bins):
            bins = np.asarray(bins, dtype=None)
        else:
            bins = np.asarray(bins)
        bins = pandas.core.reshape.tile._convert_bin_to_numeric_type(bins, dtype)
        if (np.diff(bins) < 0).any():
            raise ValueError('bins must increase monotonically.')

    labels = pandas.core.reshape.tile._format_labels(bins2, precision, right=False,
                                                     dtype=dtype)
    t = str(labels).split("\n")[0].split("(")[1][1:-2].split("),")
    for i in range(len(t) -1):
        t[i] += ")"
    t[-1] = t[-1].replace(")", "]")
    labels = Categorical(t)

    fac, bins = pandas.core.reshape.tile._bins_to_cuts(x, bins, right=right, labels=labels,
                                                       precision=precision,
                                                       include_lowest=include_lowest,
                                                       dtype=dtype,
                                                       duplicates=duplicates)
    return pandas.core.reshape.tile._postprocess_for_cut(fac, bins, retbins, x_is_series,
                                                         series_index, name, dtype)

def my_qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise'):
    x_is_series, series_index, name, x = pandas.core.reshape.tile._preprocess_for_cut(x)

    x, dtype = pandas.core.reshape.tile._coerce_to_type(x)

    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    bins = quantile(x, quantiles)

    labels = pandas.core.reshape.tile._format_labels(bins, precision, right=False,
                                                     dtype=dtype)
    bins[-1] += (bins[-1] - bins[0])*0.01
    t = str(labels).split("\n")[0].split("(")[1][1:-2].split("),")
    for i in range(len(t) -1):
        t[i] += ")"
    t[-1] = t[-1].replace(")", "]")
    labels = Categorical(t)

    fac, bins = pandas.core.reshape.tile._bins_to_cuts(x, bins, labels=labels, right=False,
                                                       precision=precision, include_lowest=True,
                                                       dtype=dtype, duplicates=duplicates)

    return pandas.core.reshape.tile._postprocess_for_cut(fac, bins, retbins, x_is_series,
                                                         series_index, name, dtype)

def _bins_to_cuts(x, bins, right=True, labels=None,
                  precision=3, include_lowest=False,
                  dtype=None, duplicates='raise'):

    if duplicates not in ['raise', 'drop']:
        raise ValueError("invalid value for 'duplicates' parameter, "
                         "valid options are: raise, drop")

    if isinstance(bins, pd.IntervalIndex):
        # we have a fast-path here
        ids = bins.get_indexer(x)
        result = take_nd(bins, ids)
        result = pd.Categorical(result, categories=bins, ordered=True)
        return result, bins
    unique_bins = unique(bins)
    if len(unique_bins) < len(bins) and len(bins) != 2:
        if duplicates == 'raise':
            raise ValueError("Bin edges must be unique: {bins!r}.\nYou "
                             "can drop duplicate edges by setting "
                             "the 'duplicates' kwarg".format(bins=bins))
        else:
            bins = unique_bins

    side = 'left' if right else 'right'
    ids = ensure_int64(bins.searchsorted(x, side=side))

    if include_lowest:
        ids[x == bins[0]] = 1

    na_mask = pd.isna(x) | (ids == len(bins)) | (ids == 0)
    has_nas = na_mask.any()

    if labels is not False:
        if labels is None:
            labels = pandas.core.reshape.tile._format_labels(bins, precision, right=right,
                                                             include_lowest=include_lowest,
                                                             dtype=dtype)
        else:
            if len(labels) != len(bins) - 1:
                raise ValueError('Bin labels must be one fewer than '
                                 'the number of bin edges')
        if not is_categorical_dtype(labels):
            labels = pd.Categorical(labels, categories=labels, ordered=True)

        np.putmask(ids, na_mask, 0)
        result = take_nd(labels, ids - 1)
    else:
        result = ids - 1
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)

    return result, bins

if __name__ == "__main__":

    jsn = {
        "id": 1,
        "type": 3,
        "typearg": [2, 3, 1, 1],
        "columns": ["sepal_lenth", "sepal_width", "petal_lenth", "petal_width"],
        "dbinfo": {
            "ip": "10.24.12.94",
            "port": 1433,
            "username": "sa",
            "password": "sa1234!",
            "dbtype": "",
            "databasename": "algonkettle",
            "inputtable": "dbo.iris",
            "outputtable": "dbo.iso"
        }
    }

    x = cut_select()
    print(x.transform_from_SQL(**jsn))
    print(vars(x))
    print("full_data" in vars(x))

