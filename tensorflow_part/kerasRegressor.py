import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


## Load the dataset
dataframe = pd.read_csv("housing.csv", delim_whitespace=True,header=None)
dataset = dataframe.values
X_train = dataset[:400,0:13]
Y_train = dataset[:400,13]
X_test = dataset[401:,0:13]
Y_test = dataset[401:,13]

##define base model
def base_model():
     model = Sequential()
     model.add(Dense(14, input_dim=13, init='normal', activation='relu'))
     model.add(Dense(7, init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model

seed = 7
np.random.seed(seed)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5,verbose=0)

clf.fit(X_test,Y_test)
res = clf.predict(X_test)

## line below throws an error
clf.score(Y_test,res)