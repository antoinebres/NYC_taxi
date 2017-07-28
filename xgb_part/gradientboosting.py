import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
def rmsle(real,predicted):
    sum=0.000
    length=len(predicted)
    for x in range(length):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/length)**0.5
TRAINING_SET_FILE = "datasets/processed_train.csv"

TEST_SET_FILE = "datasets/processed_test.csv"

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
        'pickup_hour_weekofyear',
        'pickup_pca0',
        'pickup_pca1',
        'dropoff_pca0',
        'dropoff_pca1',
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
        'day_week',
        'pickup_hour_weekofyear',
        'pickup_pca0',
        'pickup_pca1',
        'dropoff_pca0',
        'dropoff_pca1'
    ]

LABEL = "trip_duration_log"

BASE_DIR = "models"

training_set = pd.read_csv(TRAINING_SET_FILE, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

test_set = pd.read_csv(TEST_SET_FILE, skipinitialspace=True,
                            skiprows=1, names=COLUMNS)


training_set['trip_duration_log']=np.log(1+training_set['trip_duration'].values)

#test_set['trip_duration_log']=np.log(1+test_set['trip_duration'])
training_set=training_set.drop('trip_duration',axis=1)
#test_set=test_set.drop('trip_duration',axis=1)


Xtr, Xv, ytr, yv = train_test_split(training_set[FEATURES].values, training_set[LABEL].values, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dtrain1 = xgb.DMatrix(training_set[FEATURES].values, label=training_set[LABEL].values)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test_set[FEATURES].values)
watchlist = [(dtrain1, 'train')]#, (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 50, 'eta': 0.25, 'colsample_bytree': 0.3, 'max_depth': 6,
            'subsample': 0.5, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

# You could try to train with more epoch
model = xgb.train(xgb_pars, dtrain1, 200, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=20)
predtrain=model.predict(dtrain1)
for i in range(0,len(predtrain)):
	predtrain[i]=np.exp(predtrain[i])-1
predtrain1=np.exp(model.predict(dtrain1))-1
print(predtrain1-predtrain)
print(rmsle(predtrain,np.exp(training_set['trip_duration_log'].values)-1))
print(rmsle(predtrain1,np.exp(training_set['trip_duration_log'].values)-1))
#predtest=np.exp(model.predict(dtest))-1
predtest=model.predict(dtest)
for i in range(0,len(predtest)):
	predtest[i]=np.exp(predtest[i])-1
print(predtest)
COLUMNSSAM=['id','trip_duration']
sub=pd.read_csv('datasets/sample_submission.csv' ,skipinitialspace=True,
                               skiprows=1, names=COLUMNSSAM)
    #sub.drop('trip_duration',axis=1)
sub['trip_duration']=predtest
sub.to_csv('datasets/NYtaxi_26_07_18_38.csv')
#predval=model.predict(dvalid)
#print(predval)
#print(rmsle(training_set[LABEL].values,predtrain))
#print(rmsle(training_set[LABEL].values,predval))

