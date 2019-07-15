from sklearn.cluster import KMeans
from sklearn.externals import joblib


class KmeansModel:

    def __init__(self, clusters, iterations=300):
        # 必填
        self.n_clusters = clusters  # 分类数量
        self.max_iter = iterations

    def train(self, X, model_path):
        '''
        训练数据
        :return:   cluster_centers:聚类中心 ，kmeans.labels:聚类标签
        '''
        try:
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter).fit(X)

            joblib.dump(kmeans, model_path)
            return 'success'

        except Exception as e:
            print(e)
            return 'failed,{e}'.format(e=e)
