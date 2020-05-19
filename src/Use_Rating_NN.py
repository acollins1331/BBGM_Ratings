import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import pandas as pd
import math
import os # path join
from heapq import merge

FEATURES_COUNT = 24
RATINGS_COUNT = 15
chk_path = "./chk/24-2"

test_data = pd.read_csv('test.csv')

test_X = test_data.iloc[:,7:57]
test_X = test_X.drop(['BA', 'GS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OReb', 'DReb', 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'Pts', 'AtRimFG', 'LowPostFG', 'MidRangeFG', 'EWA', 'ORtg', 'DRtg', 'OWS', 'DWS', 'WS', 'WS/48'], axis=1)
test_y = test_data.iloc[:,60:75]
test_playername = test_data.iloc[:,1]

test_X = test_X.as_matrix()
test_y = test_y.as_matrix()
test_playername = test_playername.as_matrix()

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.001, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def network(stats_batch):

# Reshape input to be 1 dimensional matrix
    x_flat = tf.reshape(stats_batch, [-1, FEATURES_COUNT]) #64

# 1st layer FC
    with tf.name_scope("FC_Layer_1"):
        W_fc1 = weight_variable([FEATURES_COUNT, 1024])
        b_fc1 = bias_variable([1024])
        variable_summaries(W_fc1, name="Wfc1")
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# 2nd FC layer
    with tf.name_scope("FC_Layer_2"):
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])
        variable_summaries(W_fc2, name="Wfc2")
        variable_summaries(b_fc2, name="bfc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  
        h_fc2_drop = tf.nn.dropout(h_fc2, 1.0)

# 3rd FC layer
    with tf.name_scope("FC_Layer_3"):
        W_fc3 = weight_variable([1024, 256])
        b_fc3 = bias_variable([256])
        variable_summaries(W_fc3, name="Wfc3")
        variable_summaries(b_fc3, name="bfc3")
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, 1.0)

# Final FC layer
    with tf.name_scope("Output_Layer"):
        W_fc4 = weight_variable([256, RATINGS_COUNT])
        b_fc4 = bias_variable([RATINGS_COUNT])
        variable_summaries(W_fc4, name="Wfc4")    
        y_conv = (tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        variable_summaries(y_conv, name="Infer")
 
    return y_conv

def test():
    with tf.name_scope("Input_Function"):
        stats_batch_placeholder = tf.placeholder(tf.float64)
        ratings_batch_placeholder = tf.placeholder(tf.float64)

    with tf.name_scope("Network_Out"):
        logits_out = network(stats_batch_placeholder)

    with tf.name_scope("Loss_Nodes"):
        loss = tf.losses.mean_squared_error(labels=tf.squeeze(ratings_batch_placeholder), predictions=logits_out)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, chk_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        
        infer_out, loss_out = sess.run([logits_out, loss], 
            feed_dict={stats_batch_placeholder: test_X, ratings_batch_placeholder: test_y})

        print("loss")
        print(loss_out)
        print("player names: ")
        print(test_playername)
        print("stats: ")
        print(test_X)
        print("ratings value: ")
        print(test_y)
        print("estimated value: ")
        print(infer_out[0])

        df = pd.DataFrame(test_playername)
        tX = pd.DataFrame(test_X)
        ty = pd.DataFrame(test_y)
        io = pd.DataFrame(infer_out)
        df.insert(0, 'PID', range(1112))
        tX.insert(0, 'PID', range(1112))
        ty.insert(0, 'PID', range(1112))
        io.insert(0, 'PID', range(1112))
        df.set_index('PID')
        tX.set_index('PID')
        ty.set_index('PID')
        io.set_index('PID')
        df1 = pd.merge(df, tX, on='PID')
        df2 = pd.merge(df1, ty, on='PID')
        df3 = pd.merge(df2, io, on='PID')
        print(df3)
        df3.to_csv("foo.csv")

        coord.request_stop()
        coord.join(threads)
        sess.close()

test()
