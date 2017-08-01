import os
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.config import *


def log_trip_duration(df):
    df['trip_duration_log'] = np.log(1 + df['trip_duration'].values)
    df = df.drop('trip_duration', axis=1)
    return df


def str_model_dir(hidden_units, learning_rate):
    subdir = "hidden_units " + str(hidden_units)
    subdir += " learning_rate " + str(learning_rate)
    model_dir = "/".join([MODELS_DIR, subdir])
    return model_dir


def stack_layers(start_input, dim_in, hidden_units):
    hidden_units.insert(0, dim_in)
    next_input = start_input
    for layer_nb in range(len(hidden_units) - 2):
        layer_in = hidden_units[layer_nb]
        layer_out = hidden_units[layer_nb + 1]
        fc = fc_layer(next_input, layer_in, layer_out, "fc" + str(layer_nb))
        relu = tf.nn.relu(fc)
        next_input = relu
    return layer_out, next_input


def fc_layer(_input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(_input, w) + b
        return act


def reg_model(cv_training_set, cv_test_set, hidden_units, learning_rate):
    model_dir = str_model_dir(hidden_units, learning_rate)
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 17], name="x")
    y = tf.placeholder(tf.float32, shape=[None, ], name="y")
    layer_out, relu_out = stack_layers(x, 17, hidden_units)
    Y_pred = fc_layer(relu_out, layer_out, 1, "fc_out")
    with tf.name_scope("rmsle"):
        mean_log = tf.reduce_mean(tf.square(y-Y_pred))
        RMSLE = tf.sqrt(mean_log, name="rmsle")
        tf.summary.scalar("rmsle", RMSLE)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(RMSLE)
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_dir)
    writer.add_graph(sess.graph)

    for i in range(155):
        if i % 50 == 2:
            print()
            print("step %s out of 155" % str(i))
            s = sess.run(
                summ,
                {
                    x: cv_test_set.drop(['trip_duration_log'], axis=1).as_matrix(),
                    y:  cv_test_set['trip_duration_log'].as_matrix()
                }
            )
            writer.add_summary(s, i)
        if i % 500 == 0:
            saver.save(sess, os.path.join(model_dir, "model.ckpt"), i)
        sess.run(
            train_step,
            {
                x: cv_training_set.drop(['trip_duration_log'], axis=1).as_matrix(),
                y:  cv_training_set['trip_duration_log'].as_matrix()
            }
        )


def main(hidden_units, learning_rate):
    cv_training_set = pd.read_csv(CV_TRAINING_SET_FILE, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

    cv_test_set = pd.read_csv(CV_TEST_SET_FILE, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    cv_training_set = log_trip_duration(cv_training_set)
    cv_test_set = log_trip_duration(cv_test_set)
    print('Starting run for %s' % str_model_dir([10, 10], 0.1))

    # Actually run with the new settings
    reg_model(cv_training_set, cv_test_set, hidden_units, learning_rate)
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % MODELS_DIR)
