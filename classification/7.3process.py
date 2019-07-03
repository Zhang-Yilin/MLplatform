import pandas as pd
from sklearn import preprocessing
import Train1
import Predict1
import sklearn
from sklearn import model_selection


def Dummy(df):
    cat_col = []
    for i in df.columns:
        if df[i].dtype != 'int64' and df[i].dtype != 'float64':
            cat_col.append(i)
    cat_df = df[cat_col]
    dummy_df = pd.get_dummies(cat_df,prefix=cat_df.columns).shape
    df.drop(columns=cat_col)
    ret_df = pd.concat([dummy_df, df], axis=1)
    return ret_df


def cat(df):
    cat_col = []
    for i in df.columns:
        if df[i].dtype != 'int64' and df[i].dtype != 'float64':
            cat_col.append(i)

    combined_df = pd.concat([Dummy(pd.DataFrame(df[i])) for i in cat_col], axis=1)
    df = df.drop(columns=cat_col)
    ret_df = pd.concat([combined_df, df], axis=1)
    return ret_df

def score(y_true, y_predict):
    cm = sklearn.metrics.confusion_matrix(y_true, predict.predict_from_csv(**jsn2)[1], labels=None, sample_weight=None)
    return (sklearn.metrics.accuracy_score(y_true, y_predict), cm[0][0], cm[0][1], cm[1][0],cm[1][1])


if "__main__" == __name__:

    df = pd.read_csv('re1.csv', na_values='unknown')
    df = df.fillna(0)
    #print(df.iloc[:,:84])
    df = df.iloc[:,85:]
    features = df.columns.values.tolist()
    label = features.pop(-3)

    training_features, test_features, training_target, \
    test_target = model_selection.train_test_split(df[features], df[label] , test_size=.2, random_state=0)

    df_train = pd.concat([pd.DataFrame(training_features),pd.DataFrame(training_target)], axis = 1)
    df_train.to_csv('train.csv')
    df_predict = pd.concat([pd.DataFrame(test_features),pd.DataFrame(test_target)], axis = 1)
    df_predict.to_csv('predict.csv')


    jsn1 = {
                "model_params": {}, #参数键值， dict, eg: {"penalty": "l1"}
                "path": "train.csv", # 训练集文档存储路径，str
                "data_columns": features, # 自变量列名，list of str
                "label_columns": label, # 因变量列名 str
                "save_path": "model2.pkl" # 模型存储路径, str
               }

    jsn2 = {
                "model_params": {},
                "path": "predict.csv",
                "data_columns": features,
                "label_columns": label,
                "model_path": "model2.pkl"
               }


    train = Train1.DecisionTreeTrain()
    train.train_from_csv(**jsn1)
    logistmodel = train.get_model()

    predict = Predict1.DecisionTreePredict()
    predict.predict_from_csv(**jsn2)
    df = pd.read_csv("predict.csv")
    y_true = df[label]

    #print(sklearn.metrics.confusion_matrix(y_true, predict.predict_from_csv(**jsn2)[1], labels=None, sample_weight=None))
    print(score(y_true,predict.predict_from_csv(**jsn2)[1] ))
# logistic accuracy = 0.87119266055046
# desicion tree accuracy = 0.92587155963303

