import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV


def rmsle(real, predicted):
    _sum = 0.000
    length = len(predicted)
    for x in range(length):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        _sum = _sum + (p - r)**2
    return (_sum/length)**0.5

TRAINING_SET_FILE = "datasets/processed_train.csv"

TEST_SET_FILE = "datasets/processed_test.csv"

PREDICTION_SET_FILE = "datasets/processed_test.csv"

COLUMNS = [
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
       "manhattan_dist",
       "haversine_dist",
       "trip_to_JFK",
       "trip_from_JFK",
       "trip_from_LaGd",
       "trip_to_LaGd",
       "work",
       'trip_duration',
    ]

FEATURES = [
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
       "manhattan_dist",
       "haversine_dist",
       "trip_to_JFK",
       "trip_from_JFK",
       "trip_from_LaGd",
       "trip_to_LaGd",
       "work",
    ]

LABEL = "trip_duration_log"

BASE_DIR = "models"

training_set = pd.read_csv(TRAINING_SET_FILE, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

training_set['trip_duration_log']=np.log(1+training_set['trip_duration'].values)

#test_set['trip_duration_log']=np.log(1+test_set['trip_duration'])
training_set=training_set.drop('trip_duration',axis=1)

prediction_set = pd.read_csv(PREDICTION_SET_FILE, skipinitialspace=True,
                                  skiprows=1, names=COLUMNS)


Xtr, Xv, ytr, yv = train_test_split(training_set[FEATURES].values, training_set[LABEL].values, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(prediction_set[FEATURES].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

#############################
#Cross Validation in order to know the best hyperparameters for the problem

xgb_pars = {'min_child_weight': 50,  'colsample_bytree': 0.3,
             'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
             'eval_metric': 'rmse', 'objective': 'reg:linear'}
cv_params = {'max_depth': [4,6,8,10], 'eta': [0.1,0.2,0.25]}


model=GridSearchCV(xgb.XGBClassifier(**xgb_pars), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 



# Try different parameters! My favorite is random search :)
# xgb_pars = {'min_child_weight': 50, 'eta': 0.25, 'colsample_bytree': 0.3, 'max_depth': 10,
#             'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
#             'eval_metric': 'rmse', 'objective': 'reg:linear'}
# model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=50,
#                   maximize=False, verbose_eval=20)

#ytrain=model.predict(dtrain)
ytest = model.predict(dtest)
ytest=np.exp(ytest)-1
#print(rmsle(ytrain,training_set[LABEL].values))

COLUMNSSAM=['id','trip_duration']
sub=pd.read_csv('datasets/sample_submission.csv' ,skipinitialspace=True,
                               skiprows=1, names=COLUMNSSAM)
    #sub.drop('trip_duration',axis=1)
sub['trip_duration']=ytest
sub.to_csv('datasets/NYtaxi_27_07_12_03.csv')
