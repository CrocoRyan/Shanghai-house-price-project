import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
#模型效果评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from yellowbrick.regressor import ResidualsPlot
import pickle

from standarizer import Standarizer
from visualize import Visualizer

# FEAT_SELECTION = ['coordinate_x',
# 				 'coordinate_y', 'decoration_condition', 'deed',
# 				 'elevator', 'facility0', 'facility1', 'facility2',
# 				 'facility3', 'facility4', 'facility5', 'level',
# 				 'total', 'framework',
# 				 'ownership', 'apt', 'lift', 'district',
# 				 'rights', 'scale', 'bath', 'room',
# 				 'saloon','right_0', 'right_1', 'right_2']
FEAT_SELECTION=['decoration_condition', 'lift', 'framework', 'elevator', 'saloon', 'total', 'room', 'bath', 'scale', 'district', 'right_0', 'right_1', 'right_2', 'level', 'apt']



if __name__ == '__main__':


	train_X=pd.read_csv('./data/train_feat.csv')
	train_y=pd.read_csv('./data/train_label.csv')
	test_X=pd.read_csv('./data/test_feat.csv')
	test_y=pd.read_csv('./data/test_label.csv')

	standardizer=Standarizer(FEAT_SELECTION)
	train_X,train_y,test_X,test_y=standardizer.transform()
	train_y,test_y=train_y.ravel(),test_y.ravel()

	parameter_space = {
		'hidden_layer_sizes': [(20,), (15,), (10,)],
		'activation': ['identity', 'logistic', 'relu'],
		'solver': ['sgd', 'adam','lbfgs'],
		'alpha': [0.0001, 0.05],
		'learning_rate': ['constant', 'adaptive'],
	}
	# visualizer init
	visualizer=Visualizer(train_X,train_y,test_X,test_y)


	# random search
	# mlpr = MLPRegressor(max_iter=5000)
	# randCV = RandomizedSearchCV(estimator=mlpr, param_distributions = parameter_space,cv = 10, verbose=2, random_state=42, n_jobs = -1)
	# randCV.fit(train_X,train_y)
	# best_random = randCV.best_params_
	# best_estimator=MLPRegressor(**best_random)

	# ann
	mlpr = MLPRegressor(solver='lbfgs', alpha=1e-5,
					   hidden_layer_sizes=(15, 15, 15), random_state=1, max_iter=10000)
	# mlpr.fit(train_X[FEAT_SELECTION], train_y)


	visualizer.risidual_visualize(mlpr,"result_pictures/Residual_ann.jpg")
	filename = './models/ann_model.sav'
	mlpr.fit(train_X, train_y)
	pickle.dump(mlpr, open(filename, 'wb'))


