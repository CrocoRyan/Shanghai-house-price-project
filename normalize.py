#-*-coding:utf-8-*-
import csv
import sys
import pandas as pd
import codecs
from sklearn import preprocessing


    #
    # data = pd.read_csv('./data/data_r.csv')
    #
    # numeric_attrs = ['coordinate_x',
    #         	        'coordinate_y', 'decoration_condition','deed',
    #         	        'elevator', 'facility0', 'facility1', 'facility2',
    #         	        'facility3', 'facility4', 'facility5','level',
    #         	        'total', 'framework',
    #         	        'ownership','apt','lift','district',
    #         	        'rights','scale','bath','room',
    #         	        'saloon']
    #
    # labels=data['price']
    # tmp=data[numeric_attrs]
    # right_dummies = pd.get_dummies(data.rights)
    # right_dummies.columns=['right_0','right_1','right_2']
    #
    # data=pd.concat([tmp, right_dummies], axis=1)
    # for i in numeric_attrs:
    #     scaler = preprocessing.StandardScaler()
    #     tmp[i] = scaler.fit_transform(tmp[i].to_numpy().reshape(-1, 1))
    # labels.to_csv('./data/labels.csv',index=False)
    # tmp.to_csv('./data/data_n.csv',index=False)
class Standarizer:
    def __init__(self):
        self.train_X = pd.read_csv('./data/train_feat.csv')
        self.train_y = pd.read_csv('./data/train_label.csv')
        self.std_scale = preprocessing.StandardScaler().fit(self.train_X)
        self.std_scale_y = preprocessing.StandardScaler().fit(self.train_y)
    def transform(self):
        test_X = pd.read_csv('./data/test_feat.csv')
        test_y = pd.read_csv('./data/test_label.csv')
        X_train = self.std_scale.transform(self.train_X)
        X_test = self.std_scale.transform(test_X)
        y_train = self.std_scale_y.transform(self.train_y)
        y_test = self.std_scale_y.transform(test_y)
        return X_train,y_train,X_test,y_test
    def inverse_transform(self,labels):
        return self.std_scale_y.inverse_transform(labels)
    # X_train.to_csv('./data/train_feat.csv')
    # X_test.to_csv('./data/test_feat.csv')
    # y_train.to_csv('./data/train_label.csv')
    # y_test.to_csv('./data/test_label.csv')



