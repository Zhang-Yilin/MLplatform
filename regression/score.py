#!/usr/bin/env python
# -*- coding:utf-8 -*-

from classification.Predict1 import *


class ClassificationScore:
    def _load_data(self, **json):
        self._predata = pd.read_csv(json["path"], index_col=False)
        self._origindata = pd.read_csv(json["origin_path"], index_col=False, usecols=self._predata.columns.values.tolist())


    def RMSE(self):
        rmse = 0
        for i in range(len(self._predata)):
            rmse += (self._predata - self._origindata)**2
        rmse /= len(self._predata)
        return rmse

    def response(self, **json):
        try:
            self._load_data(**json)
            rmse = self.RMSE()
            return {"info":"success", "score": rmse}
        except Exception as e:
            return 'failed,{e}'.format(e=e)