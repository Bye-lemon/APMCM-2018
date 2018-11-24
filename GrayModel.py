# -*- coding: utf-8 -*-

from pandas import Series
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Gray_model:
    def __init__(self):
        self.a_hat = None
        self.x0 = None

    def fit(self,
            series=pd.Series(index=[1996, 1997, 1998, 1999], data=[1, 2, 3, 4])):
        """
        Series is a pd.Series with index as its date.
        :param series: pd.Series
        :return: None
        """
        self.a_hat = self._identification_algorithm(series.values)
        self.x0 = series.values[0]

    def predict(self, interval):
        result = []
        for i in range(interval):
            result.append(self.__compute(i))
        result = self.__return(result)
        return result

    def _identification_algorithm(self, series):
        B = np.array([[1] * 2] * (len(series) - 1))
        series_sum = np.cumsum(series)
        for i in range(len(series) - 1):
            B[i][0] = (series_sum[i] + series_sum[i + 1]) * (-1.0) / 2
        Y = np.transpose(series[1:])
        BT = np.transpose(B)
        a = np.linalg.inv(np.dot(BT, B))
        a = np.dot(a, BT)
        a = np.dot(a, Y)
        a = np.transpose(a)
        return a

    def score(self, series_true, series_pred, index):
        error = np.ones(len(series_true))
        relativeError = np.ones(len(series_true))
        for i in range(len(series_true)):
            error[i] = series_true[i] - series_pred[i]
            relativeError[i] = error[i] / series_pred[i] * 100
        score_record = {'GM': np.cumsum(series_pred),
                        '1—AGO': np.cumsum(series_true),
                        'Returnvalue': series_pred,
                        'Real_value': series_true,
                        'Error': error,
                        'RelativeError(%)': (relativeError)
                        }
        scores = DataFrame(score_record, index=index)
        return scores

    def __compute(self, k):
        return (self.x0 - self.a_hat[1] / self.a_hat[0]) * np.exp(-1 * self.a_hat[0] * k) + self.a_hat[1] / self.a_hat[
            0]

    def __return(self, series):
        tmp = np.ones(len(series))
        for i in range(len(series)):
            if i == 0:
                tmp[i] = series[i]
            else:
                tmp[i] = series[i] - series[i - 1]
        return tmp

    def evaluate(self, series_true, series_pred):
        scores = self.score(series_true, series_pred, np.arange(len(series_true)))

        error_square = np.dot(scores, np.transpose(scores))
        error_avg = np.mean(error_square)

        S = 0  # X0的关联度
        for i in range(1, len(series_true) - 1, 1):
            S += series_true[i] - series_true[0] + (series_pred[-1] - series_pred[0]) / 2
        S = np.abs(S)

        SK = 0  # XK的关联度
        for i in range(1, len(series_true) - 1, 1):
            SK += series_pred[i] - series_pred[0] + (series_pred[-1] - series_pred[0]) / 2
        SK = np.abs(SK)

        S_Sub = 0  # |S-SK|b
        for i in range(1, len(series_true) - 1, 1):
            S_Sub += series_true[i] - series_true[0] - (series_pred[i] - series_pred[0]) + ((series_true[-1] -
                                                                                             series_true[0]) - (
                                                                                            series_pred[i] -
                                                                                            series_pred[0])) / 2
        S_Sub = np.abs(S_Sub)

        T = (1 + S + SK) / (1 + S + SK + S_Sub)

        level = 0
        if T >= 0.9:
            level = 1
        # print ('精度为一级')
        elif T >= 0.8:
            level = 2
        # print ('精度为二级')
        elif T >= 0.7:
            level = 3
        # print ('精度为三级')
        elif T >= 0.6:
            level = 4
        # print ('精度为四级')
        return 1 - T, level

    def plot(self, series_true, series_pred, index):
        df = pd.DataFrame(index=index)
        df['Real'] = series_true
        df['Forcast'] = series_pred
        plt.figure()
        df.plot(figsize=(7, 5))
        plt.xlabel('year')
        plt.show()