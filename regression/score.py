#!/usr/bin/env python
# -*- coding:utf-8 -*-

from classification.Predict1 import *


class RegressionScore:
    def _load_data(self, **json):
        self._predata = pd.read_csv(json["path"], index_col=False)
        self._origindata = pd.read_csv(json["origin_path"], index_col=False, usecols=self._predata.columns.values.tolist())

    def RMSE(self):
        rmse = []
        for j in range(self._predata.shape[1]):
            rmsei = 0
            for i in range(len(self._predata)):
                rmsei += (self._predata.iloc[i,j] - self._origindata.iloc[i,j])**2
            rmsei /= len(self._predata)
            rmse.append(rmsei)
        return rmse

    def MAE(self):
        mae = []
        for j in range(self._predata.shape[1]):
            maei = 0
            for i in range(len(self._predata)):
                maei += abs(self._predata.iloc[i,j] - self._origindata.iloc[i,j])
            maei /= len(self._predata)
            mae.append(maei)
        return mae

    def response(self, **json):
        try:
            self._load_data(**json)
            rmse = self.RMSE()
            return {"info":"success", "score": rmse}
        except Exception as e:
            return 'failed,{e}'.format(e=e)


if __name__ == "__main__":
    json = {
        "path": "D:/pro1/response.csv",
        "origin_path": "D:/pro1/test2.csv"
    }
    n = RegressionScore()

    print(n.response(**json))
