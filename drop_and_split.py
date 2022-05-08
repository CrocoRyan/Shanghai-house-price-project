import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split



if __name__ == '__main__':
	# test_data = pd.read_csv('./data/test_data.csv')
	# test_y=test_data['price']
	# test_data=test_data.drop('price', axis=1)
	# test_data=test_data.drop('average', axis=1)
	# test_data.to_csv('./data/test_feat.csv',index=False)
	# test_y.to_csv('./data/test_label.csv',index=False)
	#
	# train_data = pd.read_csv('./data/train_data.csv')
	# train_y=train_data['price']
	# train_data=train_data.drop('price', axis=1)
	# train_data=train_data.drop('average', axis=1)
	# train_data.to_csv('./data/train_feat.csv',index=False)
	# train_y.to_csv('./data/train_label.csv',index=False)

	data = pd.read_csv('./data/data_r.csv')
	numeric_attrs = ['coordinate_x',
					 'coordinate_y', 'decoration_condition', 'deed',
					 'elevator', 'facility0', 'facility1', 'facility2',
					 'facility3', 'facility4', 'facility5', 'level',
					 'total', 'framework',
					 'ownership', 'apt', 'lift', 'district',
					 'scale', 'bath', 'room',
					 'saloon']
	y=data['price']
	tmp = data[numeric_attrs]
	right_dummies = pd.get_dummies(data.rights)
	right_dummies.columns = ['right_0', 'right_1', 'right_2']

	X = pd.concat([tmp, right_dummies], axis=1)



	X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
	X_train.to_csv('./data/train_feat.csv',index=False)
	X_test.to_csv('./data/test_feat.csv',index=False)
	y_train.to_csv('./data/train_label.csv',index=False)
	y_test.to_csv('./data/test_label.csv',index=False)

