#-*-coding:utf-8-*-
import csv
import sys
import pandas as pd
import codecs
from sklearn import preprocessing



class Standarizer:
    def __init__(self,feat_selection_list):
        self.feat_selection=feat_selection_list
        self.train_X = pd.read_csv('./data/train_feat.csv')[self.feat_selection]
        self.train_y = pd.read_csv('./data/train_label.csv')

        self.std_scale = preprocessing.StandardScaler().fit(self.train_X)
        self.std_scale_y = preprocessing.StandardScaler().fit(self.train_y)
    def transform(self):
        test_X = pd.read_csv('./data/test_feat.csv')[self.feat_selection]
        test_y = pd.read_csv('./data/test_label.csv')
        X_train = self.std_scale.transform(self.train_X)
        X_test = self.std_scale.transform(test_X[self.feat_selection])
        y_train = self.std_scale_y.transform(self.train_y)
        y_test = self.std_scale_y.transform(test_y)
        return X_train,y_train,X_test,y_test
    def inverse_transform(self,labels):
        return self.std_scale_y.inverse_transform(labels)




