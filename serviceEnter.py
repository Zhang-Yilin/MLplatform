"""
@file :serviceEnter.py
@date :2019/5/5
@author  :lizejiang
"""

from flask import Flask, abort, request, jsonify

from alg.kmeans.kmeans_test import KMeansTest
from classification.Train1 import *
from classification.Predict1 import *
from sklearn.exceptions import *

app = Flask(__name__)

tasks = []

@app.route("/", methods=["POST"])
def postTask():
    if not request.json or "id" not in request.json or "acttype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = bus(jdata=reqMess)
        return jsonify({"id": id,"info":resMess})

@app.route("/", methods=["GET"])
def getTask():
    if not request.json or "id" not in request.json or "acttype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = bus(jdata=reqMess)
        return jsonify({"id": id,"info":resMess})

@app.route("/classificationtest/", methods=["POST"])
def postTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = classification(jdata=reqMess)
        return jsonify({"id": id,"info": resMess})

@app.route("/classificationtest/", methods=["GET"])
def getTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = classificationPredict(jdata=reqMess)
        return jsonify({"id": id,"info":resMess["info"], "value": str(resMess["pre_value"])})


def bus(jdata):
    algtype = jdata["algtype"]
    acttype = jdata["acttype"]
    reMess = ""
    # 0是训练，1是预测
    if "0" == acttype:
        if "kmeans"==algtype:
            kMeansTest = KMeansTest()
            reMess = kMeansTest.train_section(jdata)
        else:
            reMess = "failed,algtype must be an algorithm type"
    elif "1" == acttype:
        if "kmeans" == algtype:
            kMeansTest = KMeansTest()
            reMess = kMeansTest.predict_section(jdata)
        else:
            reMess = "failed,algtype must be an algorithm type"
    else:
        reMess = "failed,acttype must be in 0 or 1,0 means training,1 means prediction"
    return reMess

def classification(jdata):
    algtype = jdata["algtype"]
    if algtype == "decisionTree":
        model = ClassificationTrain()
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            reMess = model.train_from_csv(**jdata)
    elif algtype == "logistic":
        model = LogisticRegressionTrain()
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            reMess = model.train_from_csv(**jdata)
    else:
        reMess = "failed, algtype must be an algorithm type"
    return reMess

def classificationPredict(jdata):
    algtype = jdata["algtype"]
    if algtype == "decisionTree":
        model = ClassificationPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "logistic":
        model = LogisticRegressionPredict()
        reMess = model.predict_from_csv(**jdata)
    else:
        reMess = "failed, algtype must be an algorithm type"
    return reMess


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=6666, debug=True)  #, debug=True



"""
入参格式
{
	"id": 1,
	"algtype":"kmeans",
	"acttype":"0",
	"iterations": 200,
	"clusters": 10,
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

分类预测入参：
{
            "id":1,
            "algtype":"logistic",
            "model_params": {"max_iter":100},
            "path": "D:/pro1/test2.csv",
            "data_columns": ["组织组", "总金额(元)", "销售组织单位", "运算加权总金额(元)", "客户类型", "客户级别",
                             "一级行业", "二级行业", "三级行业", "浪潮所属行业部", "行业部二级部门", "城市",
                             "预计招标类型", "项目公司1属性", "项目级别", "获得途径", "是否指名客户", "客户预算来源",
                             "客户预算", "商机产生背景", "采购特点", "技术方案是否已有", "是否存在竞争对手",
                             "评标环节可控制", "合作渠道招标方关系好", "用户或采购方表态支持", "标书指标倾向浪潮",
                             "总体规划方案已有", "项目招标方案已有", "是否老客户", "负责员工职位"],
            "label_columns": ["是否中标"],
            "save_path": "D:/pro1/classification/model.pkl"
}


分类预测入参：
{
            "id":1,
            "algtype":"logistic",
            "path": "D:/pro1/test2.csv",
            "model_path": "D:/pro1/model1.pkl",
            "save_path": "D:/pro1/response.csv"
}
"""