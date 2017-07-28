import pandas as pd
#import tensorflow as tf
import numpy as np
from sklearn import linear_model
reg=linear_model.Lasso(alpha=0.3)

COLUMNS = [
        "manhattan_dist",
        "haversine_dist",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        'passenger_count',
        'vendor_id',
        'store_and_fwd_flag',
        'pickup_hour',
        'day_week',
        'trip_duration'
    ]

FEATURES = [
        "manhattan_dist",
        "haversine_dist",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        'passenger_count',
        'vendor_id',
        'store_and_fwd_flag',
        'pickup_hour',
        'day_week'
    ]


df_train=pd.read_csv("datasets/processed_train.csv",skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
df_test=pd.read_csv("datasets/processed_train.csv",skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

#FEATURES = ['passenger_count', 'vendor_id', 'store_and_fwd_flag',
            #'pickup_hour', 'day_week']

LABEL = "trip_duration"

X=df_train.drop('trip_duration',axis=1)
Xtest=df_test.drop('trip_duration',axis=1)
y=df_train.ix[:,"trip_duration"]
ytest=df_test.ix[:,"trip_duration"]
reg.fit(X,y)
score=reg.score(X,y)
pred=reg.predict(Xtest)
def rmsle(real,predicted):
    sum=0.000
    length=len(predicted)
    for x in range(length):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/length)**0.5
dt=rmsle(pred,ytest)
print(score)
print(dt)
from sklearn.ensemble import RandomForestRegressor
max_depth=20
reg2= RandomForestRegressor(max_depth=max_depth, random_state=2)
reg2.fit(X,y)
pred2=reg2.predict(Xtest)
score2=reg2.score(X,y)
dt2=rmsle(pred2,ytest)
print(score2)
print(dt2)