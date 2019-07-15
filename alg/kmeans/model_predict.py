from sklearn.externals import joblib


class ModelPredict:
    def predict(self, pre_data, model_path):
        model = joblib.load(model_path)
        return model.predict(pre_data)
