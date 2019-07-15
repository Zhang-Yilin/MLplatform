from alg.kmeans.model_predict import ModelPredict
from alg.kmeans.model_train import KmeansModel
from sqlcon.sqlserver.sqlserver import SQLServer

'''数据库取数'''


class KMeansTest:
    # 训练部分
    def train_section(self, json_file):
        #json_dict = json.loads(json_file)
        json_dict = json_file
        sQLServer = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'], user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])
        train_data = sQLServer.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'], cols=json_dict['columns'])
        k_means_model = KmeansModel(clusters=json_dict['clusters'], iterations=json_dict['iterations'])
        ret = k_means_model.train(train_data, 'alg/kmeans/kmeans_model.pkl')
        return ret

    '''预测部分'''

    def predict_section(self, json_file):
        #json_dict = json.loads(json_file)
        json_dict = json_file
        sQLServer = SQLServer(host=json_dict['dbinfo']['ip'], port=json_dict['dbinfo']['port'], user=json_dict['dbinfo']['username'],
                              pwd=json_dict['dbinfo']['password'], db=json_dict['dbinfo']['databasename'])
        pre_data = sQLServer.df_read_sqlserver(table=json_dict['dbinfo']['inputtable'], cols=json_dict['columns'])
        modelPredict = ModelPredict()
        pre = modelPredict.predict(pre_data, 'alg/kmeans/kmeans_model.pkl')
        pre_data['class'] = pre
        #结果生成csv
        pre_data.to_csv('alg/kmeans/kmeans_data.csv', index=0)
        #结果存入数据库
        rewrite=sQLServer.df_write_sqlserver(table=json_dict['dbinfo']['outputtable'], df=pre_data, cols=json_dict['columns'])
        return rewrite


if __name__ == '__main__':
    kMeansTest = KMeansTest()
    jsn = {"id": 1, "iterations": 200, "clusters": 3, "columns": ["sepal_lenth","sepal_width","petal_lenth","petal_width"], "dbinfo": {"ip": "10.24.12.94", "port": 1433, "username": "sa", "password": "sa1234!", "dbtype": "", "databasename": "algonkettle", "inputtable": "dbo.iris","outputtable": "dbo.iso"}}
    print(kMeansTest.train_section(jsn))
    print(kMeansTest.predict_section(jsn))
