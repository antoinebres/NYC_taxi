# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer


def next_batch(num, data, labels, predo):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    pred_shuffle=[predo[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(pred_shuffle)


TRAINING_SET_FILE2 = "datasets/trainingNN1.csv"
COLUMNS = [
       'id',
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
       "total_distance",
       "total_duration",
       "number_of_streets",
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
       "total_distance",
       "total_duration",
       "number_of_streets",
    ]

LABEL = "trip_duration_log"

BASE_DIR = "models"

training_set2 = pd.read_csv(TRAINING_SET_FILE2, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

training_set2['trip_duration_log']=np.log(1+training_set2['trip_duration'].values)

training_set2=training_set2.drop('trip_duration',axis=1)


training_set2pred=pd.read_csv('datasets/trainingNN1pred.csv')
training_set2pred_others=pd.read_csv('datasets/trainingNN1pred_others.csv')
training_set2pred = training_set2pred.merge(training_set2pred_others, left_on='id', right_on='RFR0', how='outer')
training_set2pred = training_set2pred.drop(['id_y'],axis=1)


# Variables
nb_inputs=20
nb_models=71
learning_rate=0.003

with tf.device("/gpu:1"):
  # Inputs
  X = tf.placeholder(tf.float32, [None, nb_inputs])
  # Correct labels
  Y_ = tf.placeholder(tf.float32, [None, ])
  # Prédictions des modèles
  pred = tf.placeholder(tf.float32, [None, nb_models])


  # five layers and their number of neurons (tha last layer has 10 softmax neurons)
  L = 100
  M = 60
  N = nb_models
  #O = 2

  # Weights initialised with small random values between -0.2 and +0.2
  # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
  pkeep = tf.placeholder(tf.float32)
  W1 = tf.Variable(tf.truncated_normal([nb_inputs, L], stddev=0.1))
  B1 = tf.Variable(tf.ones([L])/10)
  W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
  B2 = tf.Variable(tf.ones([M])/10)
  W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
  B3 = tf.Variable(tf.ones([N])/10)

  Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
  Y1d = tf.nn.dropout(Y1, pkeep)
  Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
  Ylogits = tf.matmul(Y2, W3) + B3
  Y = tf.nn.softmax(Ylogits)

  fusion = tf.reduce_sum(tf.multiply(Y,pred), 1)
  mean_log= tf.reduce_mean(tf.square(fusion-Y_))
  RMS=tf.sqrt(mean_log)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_log)

init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config = config)
sess.run(init)



rmsle_train=[]
rmsle_test=[]
# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    
    # compute training values for visualisation
    if update_train_data:
        rms = sess.run([RMS], {X: training_set2.drop(['trip_duration_log','id'],axis=1).as_matrix(), Y_:  training_set2['trip_duration_log'].as_matrix(), pred: training_set2pred.drop(['id_x'],axis=1).as_matrix(), pkeep:1})
        print(str(i) + ": RMSLE:" + str(rms))
        rmsle_train.append(rms)
        #datavis.append_training_curves_data(i, a, c)
        #datavis.update_image1(im)
        #datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    #if update_test_data:
        #rms = sess.run([RMS], {X: xtrain[2300:2333], Y_: ytrain[2300:2333],pkeep:1})
        #print(str(i) + ": RMSLE:" + str(rms))
        #rsmle_test.append(rms)
        #print(str(i) + ": accuracyval:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        #datavis.append_test_curves_data(i, a, c)
        #datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pred: batch_pred, pkeep:0.5})
    return(rmsle_train)


for i in range(0,10):
    # batch_X,batch_Y,batch_pred = next_batch(200, training_set2.drop(['trip_duration_log','id'],axis=1).as_matrix(), training_set2['trip_duration_log'].as_matrix(), training_set2pred.drop(['id_x'],axis=1).as_matrix() )
    training_step(i,i%50==0,i%50==0)


PREDICTION_SET_FILE = "datasets/processed_testNEW.csv"
PREDICTION_FILE = "datasets/processed_testNEWpred.csv"
PREDICTION_FILE_OTHERS = "datasets/processed_testNEWpred_others.csv"
COLUMNSSAM=['id','trip_duration']
prediction_test= pd.read_csv(PREDICTION_FILE)
prediction_test_others= pd.read_csv(PREDICTION_FILE_OTHERS)
prediction_test = prediction_test.merge(prediction_test_others, left_on='id', right_on='RFR0', how='outer').training_set2pred.drop(['id_y'],axis=1)
sub=pd.read_csv('datasets/sample_submission.csv' ,skipinitialspace=True,
                               skiprows=1, names=COLUMNSSAM)
prediction_set = pd.read_csv(PREDICTION_SET_FILE, skipinitialspace=True,
                                  skiprows=1, names=COLUMNS)
    #sub.drop('trip_duration',axis=1)
ytest=sess.run([fusion], {X: prediction_set.drop(['trip_duration','id'],axis=1).as_matrix(), pkeep:1, pred: prediction_test.drop(['id_x'],axis=1).as_matrix()})
print(np.array(ytest).shape)
sub['trip_duration']=np.transpose(np.array(ytest))
sub.to_csv('datasets/NYtaxi_31_07_17_23.csv')




