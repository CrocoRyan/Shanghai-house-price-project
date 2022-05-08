import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import pickle
from standarizer import Standarizer
from visualize import Visualizer

FEAT_SELECTIONS=['decoration_condition', 'lift', 'framework', 'elevator', 'saloon', 'total', 'room', 'bath', 'scale','district','right_0','right_1','right_2','level','apt']

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 40, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

train_X = pd.read_csv('./data/train_feat.csv')
train_y = pd.read_csv('./data/train_label.csv')
test_X = pd.read_csv('./data/test_feat.csv')
test_y = pd.read_csv('./data/test_label.csv')

standardizer = Standarizer(FEAT_SELECTIONS)
train_X, train_y, test_X, test_y = standardizer.transform()
train_y, test_y = train_y.ravel(), test_y.ravel()



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 5, verbose=2, n_jobs = -1)

# Fit the random search model
rf_random.fit(train_X, train_y)

visualizer = Visualizer(train_X, train_y, test_X, test_y)
base_model = RandomForestRegressor(max_depth=10, random_state=42)
base_model.fit(train_X, train_y)

best_random = rf_random.best_params_
best_estimator=RandomForestRegressor(**best_random)
visualizer.risidual_visualize(base_model, "result_pictures/Residual_random_forest_base.jpg")
visualizer.risidual_visualize(best_estimator, "result_pictures/Residual_random_forest_best.jpg")
filename = './models/random_forest_model.sav'
best_estimator.fit(train_X,train_y)
pickle.dump(best_estimator, open(filename, 'wb'))





