
# coding: utf-8

# In[1]:

#Imports
import pandas as pd
import numpy as np
import tensorflow as tf
import random


# In[2]:

#Helper functions
def one_hot(i):
    one_hot = [0 for _ in range(3)]
    one_hot[i] = 1
    return one_hot


# In[3]:

#Read data
dataset = pd.read_csv('wine.csv')
col_names = list(dataset.columns.values)
features = col_names[1:]
#Normalize data
for f in features:
    dataset[f] = (dataset[f] - dataset[f].mean())/dataset[f].std()
#Split dataset
size = dataset.shape[0]
training_size = int(0.7*size)
shuffled_dataset = dataset.sample(frac=1)
labels = dataset[features]
training_sample = shuffled_dataset.iloc[:training_size]
training_set = training_sample[features].values.astype(np.float32)
training_labels = np.array((training_sample['Class'] - 1).apply(one_hot).values.tolist()).astype(np.float32)
test_sample = shuffled_dataset.iloc[training_size:]
test_set = test_sample[features].values
test_labels = np.array((test_sample['Class'] - 1).apply(one_hot).values.tolist()).astype(np.float32)


# In[4]:

#TENSORFLOW
#Variable
X = tf.placeholder(tf.float32, [None, 13], name='input')
Y_ = tf.placeholder(tf.float32, [None, 3], name='label')

with tf.name_scope('linear_classifier'):
    #Weights and biases
    W = tf.Variable(tf.truncated_normal([13,3], stddev=0.1), name='W')
    b = tf.Variable(tf.zeros([3])+0.1, name='b')
    #Model
    logits = tf.matmul(X,W)+b
    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)
#Accuracy
with tf.name_scope('cross_entropy'):
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_), name = 'cross_entropy')
    tf.summary.scalar('cross_entropy', xent)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
#Train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(xent)
#Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#Training phase
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('tmp/Wine')
writer.add_graph(sess.graph)


# In[5]:

for i in range(10000):
    s = sess.run(merged_summary, {X:training_set, Y_: training_labels})
    writer.add_summary(s, i)
    sess.run(train_step, {X:training_set, Y_: training_labels})
acc = sess.run(accuracy, {X:training_set, Y_: training_labels})
ac = sess.run(accuracy, {X:test_set, Y_: test_labels})
print(acc, ac)

