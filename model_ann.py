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

from normalize import Standarizer
from visualize import Visualizer


FEAT_SELECTION=['decoration_condition', 'lift', 'framework', 'elevator', 'saloon', 'total', 'room', 'bath', 'scale', 'price']



if __name__ == '__main__':


	train_X=pd.read_csv('./data/train_feat.csv')
	train_y=pd.read_csv('./data/train_label.csv')
	test_X=pd.read_csv('./data/test_feat.csv')
	test_y=pd.read_csv('./data/test_label.csv')

	standardizer=Standarizer()
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


	# ann
	mlpr = MLPRegressor(max_iter=5000)
	randCV = RandomizedSearchCV(estimator=mlpr, param_distributions = parameter_space,cv = 5, verbose=2, random_state=42, n_jobs = -1)
	randCV.fit(train_X,train_y)
	best_random = randCV.best_params_
	best_estimator=MLPRegressor(**best_random)

	visualizer.risidual_visualize(best_estimator,"result_pictures/Residual_ann.jpg")



