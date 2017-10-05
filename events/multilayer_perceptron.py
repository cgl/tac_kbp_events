""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


from __future__ import print_function

import tensorflow as tf
import numpy as np
import datetime
from sequence_detection import after_links_as_dictionary,write_results_tbf
# Parameters
learning_rate = 0.001
training_epochs = 350
batch_size = 100
display_step = 25

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 2nd layer number of neurons
n_input = 912 #  data input
n_classes = 2 #  total classes

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def one_hot_y(list_of_bin):
    """
    Takes list of binary input and returns one hot representation as numpy array
    """
    return np.column_stack((1 - np.array(list_of_bin),np.array(list_of_bin)))

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

from sequence_detection import get_dataset
X_train,y_train,IDS,_ = get_dataset("data/LDC2016E130_training.tbf",training=True)
X_test,y_test,IDS_test,events = get_dataset("data/LDC2016E130_test.tbf",training=False)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = np.array(X_train[i*batch_size:(i*batch_size)+batch_size])
            batch_y = one_hot_y(y_train[i*batch_size:(i*batch_size)+batch_size])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    y_pred = tf.argmax(pred, 1)
    y_true = tf.argmax(Y, 1)
    correct_prediction = tf.equal(y_pred, tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    TP = tf.count_nonzero(y_pred * y_true, dtype=tf.float32)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1), dtype=tf.float32)
    FP = tf.count_nonzero(y_pred * (y_true - 1), dtype=tf.float32)
    FN = tf.count_nonzero((y_pred - 1) * y_true, dtype=tf.float32)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    val_accuracy,precision,recall,f1, y_pred = sess.run([accuracy, precision,recall,f1, y_pred], feed_dict={X: np.array(X_test), Y: one_hot_y(y_test)})

    print("Accuracy:", val_accuracy)
    print("Results:%s\t%s\t%s\n" %(precision,recall,f1))

    afters_pred =  after_links_as_dictionary(y_pred,IDS_test,events)
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    write_results_tbf(events, afters_pred,run_id="%s-%s" %("Mlp-3Layer",timestamp))
