#!/usr/bin/env python
# -*- coding:utf-8 -*-

from classification.Predict1 import *
from sklearn.metrics import confusion_matrix, accuracy_score

class ClassificationScore:
    def _load_data(self, **json):
        self._predata = pd.read_csv(json["path"], index_col=False)
        self._origindata = pd.read_csv(json["origin_path"], index_col=False, usecols=self._predata.columns.values.tolist())

    def con_matrix(self):
        cm = confusion_matrix(y_true=self._origindata, y_pred=self._predata, labels=None, sample_weight=None)
        return cm

    def score(self):
        accuracyScore = accuracy_score(y_true=self._origindata, y_pred=self._predata)
        return accuracyScore

    def response(self, **json):
        self._load_data(**json)
        cmlist = []
        cm = self.con_matrix()
        print(cm)
        for i in cm:
            cmlist = cmlist + list(i)
        return {"score": self.score(), "confusion_matrix": cmlist}



if __name__ == "__main__":
    json = {
        "path": "D:/pro1/response.csv",
        "origin_path": "D:/pro1/test2.csv"
    }
    n = ClassificationScore()

    print(n.response(**json))
