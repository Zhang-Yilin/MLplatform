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
from regression.regressionTrain import *
from regression.regressionPredict import *
from classification.score import ClassificationScore
from regression.score import RegressionScore

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

@app.route("/classification/", methods=["POST"])
def postTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = classification(jdata=reqMess)
        return jsonify({"id": id,"info": resMess})

@app.route("/classification/", methods=["GET"])
def getTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = classificationPredict(jdata=reqMess)
        return jsonify({"id": id,"info":resMess["info"], "value": str(resMess["pre_value"])})

@app.route("/regression/", methods=["POST"])
def postTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = regression(jdata=reqMess)
        return jsonify({"id": id,"info": resMess})

@app.route("/regression/", methods=["GET"])
def getTest():
    if not request.json or "id" not in request.json or "algtype" not in request.json:
        abort(400)
    else:
        reqMess = request.json
        print(reqMess)
        id = reqMess["id"]
        resMess = regressionPredict(jdata=reqMess)
        return jsonify({"id": id,"info":resMess["info"], "value": str(resMess["pre_value"])})

@app.route("/classificationscore/", methods=["POST"])
def scoreTask():
    if not request.json or "id" not in request.json:
        abort(400)
    else:
        reqMess = request
        id = reqMess["id"]
        resMess = getscore(jdata=reqMess)
        return jsonify({"id": id, "info": resMess["info"], "accuracy": resMess["score"]})

@app.route("/classificationscore/", methods=["POST"])
def scoreTask():
    if not request.json or "id" not in request.json:
        abort(400)
    else:
        reqMess = request
        id = reqMess["id"]
        resMess =regressionScore(jdata=reqMess)
        return jsonify({"id": id, "info": resMess["info"], "accuracy": resMess["score"]})


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
    algtype = jdata["algtype"].lower()
    reMess = ""
    if algtype == "decisiontree":
        model = DecisionTreeTrain()
    elif algtype == "logistic":
        model = LogisticRegressionTrain()
    elif algtype == "randomforest":
        model = RandomForestTrain()
    elif algtype == "gradientboosting":
        model = GradientBoostingTrain()
    elif algtype == "mlp" or algtype == "neuronnetwork":
        model = MLPTrain()
    elif algtype == "lineardiscriminantanalysis" or algtype == "lda":
        model = LinearDiscriminantAnalysisTrain()
    elif algtype == "svm":
        model = SVMTrain()
    elif algtype == "knn":
        model = KNNTrain()
    elif algtype == "xgradientboosting" or algtype == "xgboosting":
        model = XGradientBoostingTrain()
    elif algtype == "adaboost":
        model = AdaBoostTrain()
    else:
        reMess = "failed, algtype must be an algorithm type"
    if "failed, algtype must be an algorithm type" != reMess:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            reMess = model.train_from_csv(**jdata)
    return reMess

def classificationPredict(jdata):
    algtype = jdata["algtype"]
    if algtype == "decisionTree":
        model = DecisionTreePredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "logistic":
        model = LogisticRegressionPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "randomforest":
        model = RandomForestPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "gradientboosting":
        model = GradientBoostingPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "mlp" or algtype == "neuronnetwork":
        model = MLPPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "lineardiscriminantanalysis" or algtype == "lda":
        model = LinearDiscriminantAnalysisPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "svm":
        model = SVMPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "knn":
        model = KNNPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "xgradientboosting" or algtype == "xgboosting":
        model = XGradientBoostingPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "adaboost":
        model = AdaBoostPredict()
        reMess = model.predict_from_csv(**jdata)
    else:
        reMess = "failed, algtype must be an algorithm type"
    return reMess

def regression(jdata):
    algtype = jdata["algtype"].lower()
    reMess = ""
    if algtype == "svm" or algtype == "svr":
        model = SVRTrain()
    elif algtype == "linearregression":
        model = LinearRegressionTrain()
    elif algtype == "decisiontreeregressor":
        model = DecisionTreeRegressorTrain()
    elif algtype == "gradientboostingregressor":
        model =GradientBoostingRegressorTrain()
    elif algtype == "mlpregressor" or algtype == "neuronnetworkregressor":
        model = MLPRegressorTrain()
    elif algtype == "isotonicregression":
        model = IsotonicRegressionTrain()
    elif algtype == "polynomialregression":
        model = PolynomialRegressionTrain()
    elif algtype == "randomforestregressor":
        model = RandomForestRegressorTrain()
    else:
        reMess = "failed, algtype must be an algorithm type"
    if "failed, algtype must be an algorithm type" != reMess:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            reMess = model.train_from_csv(**jdata)
    return reMess

def regressionPredict(jdata):
    algtype = jdata["algtype"]
    if algtype == "svm" or algtype == "svr":
        model = SVRPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "linearregression":
        model = LinearRegressionPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "gradientboostingregressor":
        model = GradientBoostingRegressorPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "decisiontreeregressor":
        model = DecisionTreeRegressorPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "mlpregressor" or algtype == "neuronnetworkregressor":
        model = MLPRegressorPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "isotonicregression":
        model = IsotonicRegressionPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "polynomialregression":
        model = PolynomialRegressionPredict()
        reMess = model.predict_from_csv(**jdata)
    elif algtype == "randomforestregressor":
        model = RandomForestRegressorPredict()
        reMess = model.predict_from_csv(**jdata)
    else:
        reMess = "failed, algtype must be an algorithm type"
    return reMess

def getscore(jdata):
    scoremodel = ClassificationScore()
    reMess = scoremodel.response(**jdata)
    return reMess

def regressionScore(jdata):
    scoremodel = RegressionScore()
    reMess = scoremodel.response(**jdata)
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

分类,回归训练入参：
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


分类，回归预测入参：
{
            "id":1,
            "algtype":"logistic",
            "path": "D:/pro1/test2.csv",
            "model_path": "D:/pro1/model1.pkl",
            "save_path": "D:/pro1/response.csv"
}

分类，回归打分入参
{
        "id":1
        "path": "D:/pro1/response.csv",
        "origin_path": "D:/pro1/test2.csv"
}
"""