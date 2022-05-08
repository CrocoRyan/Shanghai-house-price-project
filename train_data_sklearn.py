import pickle

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso


from standarizer import Standarizer
from visualize import Visualizer


FEAT_SELECTION=['decoration_condition', 'lift', 'framework', 'elevator', 'saloon', 'total', 'room', 'bath', 'scale', 'district', 'right_0', 'right_1', 'right_2', 'level', 'apt']
if __name__ == '__main__':


	train_X=pd.read_csv('./data/train_feat.csv')
	train_y=pd.read_csv('./data/train_label.csv')
	test_X=pd.read_csv('./data/test_feat.csv')
	test_y=pd.read_csv('./data/test_label.csv')

	standardizer=Standarizer(FEAT_SELECTION)
	train_X,train_y,test_X,test_y=standardizer.transform()
	train_y,test_y=train_y.ravel(),test_y.ravel()


	# visualizer init
	visualizer=Visualizer(train_X,train_y,test_X,test_y)

	#linear
	line = LinearRegression()
	line.fit(train_X,train_y)
	visualizer.risidual_visualize(line,"result_pictures/Residual_linear.jpg")
	filename = './models/line_model.sav'
	line.fit(train_X, train_y)
	pickle.dump(line, open(filename, 'wb'))

