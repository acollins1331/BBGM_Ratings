import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import pandas as pd
import math
import os # path join
from heapq import merge

FEATURES_COUNT = 31
RATINGS_COUNT = 15
learning_rate = .00001
logs_path = "./logs/31_alt"
val_logs_path = "./val_logs/31_alt"
chk_path = "./chk/31_alt"
best_path = "./best_loss_alt.txt"

train_data = pd.read_csv('train720.csv')
val_data = pd.read_csv('validation720.csv')

train_X = train_data.iloc[:,7:58]
train_X = train_X.drop(['GP', 'GS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OReb', 'DReb', 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'Pts', 'AtRimFG', 'LowPostFG', 'MidRangeFG', 'WS'], axis=1)
train_y = train_data.iloc[:,60:75]
#train_y = train_y.drop(['ENDU'], axis=1)
train_playername = train_data.iloc[:,1]

train_X = train_X.as_matrix()
train_y = train_y.as_matrix()
train_playername = train_playername.as_matrix()

val_X = val_data.iloc[:,7:58]
val_X = val_X.drop(['GP', 'GS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OReb', 'DReb', 'Reb', 'Ast', 'TO', 'Stl', 'Blk', 'Pts', 'AtRimFG', 'LowPostFG', 'MidRangeFG', 'WS'], axis=1)
val_y = val_data.iloc[:,60:75]
#val_y = val_y.drop(['ENDU'], axis=1)
val_playername = val_data.iloc[:,1]

val_X = val_X.as_matrix()
val_y = val_y.as_matrix()
val_playername = val_playername.as_matrix()

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
        W_fc2 = weight_variable([1024, 2048])
        b_fc2 = bias_variable([2048])
        variable_summaries(W_fc2, name="Wfc2")
        variable_summaries(b_fc2, name="bfc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  
        h_fc2_drop = tf.nn.dropout(h_fc2, 1.0)

# 3rd FC layer
    with tf.name_scope("FC_Layer_3"):
        W_fc3 = weight_variable([2048, 1024])
        b_fc3 = bias_variable([1024])
        variable_summaries(W_fc3, name="Wfc3")
        variable_summaries(b_fc3, name="bfc3")
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, 1.0)

# 4th FC layer
    with tf.name_scope("FC_Layer_4"):
        W_fc4 = weight_variable([1024, 256])
        b_fc4 = bias_variable([256])
        variable_summaries(W_fc4, name="Wfc4")
        variable_summaries(b_fc4, name="bfc4")
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, 1.0)

# Output layer
    with tf.name_scope("Output_Layer"):
        W_fc5 = weight_variable([256, RATINGS_COUNT])
        b_fc5= bias_variable([RATINGS_COUNT])
        variable_summaries(W_fc5, name="Wfc5")    
        y_conv = (tf.matmul(h_fc4_drop, W_fc5) + b_fc5)
        variable_summaries(y_conv, name="Infer")
 
    return y_conv

def train():
    with tf.name_scope("Input_Function"):
        stats_batch_placeholder = tf.placeholder(tf.float64)
        ratings_batch_placeholder = tf.placeholder(tf.float64)

    with tf.name_scope("Network_Out"):
        logits_out = network(stats_batch_placeholder)

    with tf.name_scope("Loss_Nodes"):
        loss = tf.losses.mean_squared_error(labels=tf.squeeze(ratings_batch_placeholder), predictions=logits_out)
    with tf.name_scope("Step_Nodes"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope("Summary_Nodes"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("learning rate", learning_rate)
        merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        val_summary_writer = tf.summary.FileWriter(val_logs_path, graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if(tf.train.checkpoint_exists(chk_path)):
            saver.restore(sess, chk_path)
        else:
            saver.save(sess, chk_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        
        loss_file = open(best_path, 'r')
        best_loss = float(loss_file.read())
        loss_file.close()

        improved = 0
        not_improved = 0

        for i in range(1000000):
            _, infer_out, loss_out, summary, global_step_out = sess.run([train_step, logits_out, loss, merged_summary_op, global_step], 
                feed_dict={stats_batch_placeholder: train_X, ratings_batch_placeholder: train_y})

            print("loss: " + str(loss_out))
            print("improved: " + str(improved))
            print("not improved: " + str(not_improved))
            print("val loss: " + str(best_loss))
            print("global step: " + str(global_step_out - 1))
            
            print("player names: ")
            for j in range(0, 5):
                print(train_playername[j])
            #print("stats: ")
            #for j in range(0, 5):
                #print(train_X[j])
            print("ratings value: ")
            for j in range(0, 5):
                print(train_y[j])
            print("estimated value: ")
            for j in range(0, 5):
                print((infer_out[j].astype(int)))

            #write summary every 100 iterations to view in Tensorboard
            if(i%100 == 0):
                summary_writer.add_summary(summary, global_step_out) 
            #check loss of validation set on current model
            if(i%250 == 0 and i != 0):
                val_infer_out, val_loss_out, val_summary = sess.run([logits_out, loss, merged_summary_op], 
                    feed_dict={stats_batch_placeholder: val_X, ratings_batch_placeholder: val_y})
                print("val loss: " + str(val_loss_out))
                print("player names: ")
                for j in range(0, 5):
                    print(val_playername[j])
                #print("stats: ")
                #for j in range(0, 5):
                    #print(train_X[j])
                print("ratings value: ")
                for j in range(0, 5):
                    print(val_y[j])
                print("estimated value: ")
                for j in range(0, 5):
                    print((infer_out[j].astype(int)))
                val_summary_writer.add_summary(val_summary, global_step_out) 
                #if the loss is better than it has been before
                #save the model
                if(val_loss_out < best_loss):
                    print(str(val_loss_out) + " vs " + str(best_loss))    
                    print("New best loss! saved!") 
                    loss_file = open(best_path, 'w')
                    loss_file.write(str(val_loss_out))
                    loss_file.close() 
                    saver.save(sess, chk_path)
                    best_loss = val_loss_out
                    improved += 1
                else:
                    print(str(val_loss_out) + " vs " + str(best_loss))
                    print("Not new best loss! not saved!")
                    not_improved += 1

        coord.request_stop()
        coord.join(threads)
        sess.close()

train()
