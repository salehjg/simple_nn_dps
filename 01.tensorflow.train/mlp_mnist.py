""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function
import tensorflow as tf
import mnist as dataset
import numpy as np


# convert labels to one-hot encoding
def batch_make_onehot(batch_labels, nclasses):
    rslts = np.zeros(shape=[batch_labels.shape[0], nclasses])
    for i in range(batch_labels.shape[0]):
        rslts[i, :] = make_onehot(batch_labels[i], nclasses)
    return rslts

def make_onehot(label, nclasses):
    return np.eye(nclasses)[label]


dataset.init()
x_train, t_train, x_test, t_test = dataset.load()
t_train = batch_make_onehot(t_train, 10)
t_test = batch_make_onehot(t_test, 10)

np.save("data/dataset/mnist_train_data_60000x784.npy", x_train)
np.save("data/dataset/mnist_train_label_60000x10.npy", t_train)
np.save("data/dataset/mnist_test_data_10000x10.npy", x_test)
np.save("data/dataset/mnist_test_label_10000x10.npy", t_test)

# Parameters
learning_rate = 0.001
training_epochs = 5
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
    'hout': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='hout')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'bout': tf.Variable(tf.random_normal([n_classes]), name='bout')
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['hout']) + biases['bout']
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

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = x_train[i * batch_size: (i + 1) * batch_size, :], t_train[
                                                                                 i * batch_size: (i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_test, Y: t_test}))

    # Weight Extraction
    trained = tf.trainable_variables()

    trained_h1 = [v for v in trained if v.name == "h1:0"][0]
    trained_h2 = [v for v in trained if v.name == "h2:0"][0]
    trained_hout = [v for v in trained if v.name == "hout:0"][0]

    trained_b1 = [v for v in trained if v.name == "b1:0"][0]
    trained_b2 = [v for v in trained if v.name == "b2:0"][0]
    trained_bout = [v for v in trained if v.name == "bout:0"][0]

    trained_h1, trained_h2, trained_hout, trained_b1, trained_b2, trained_bout = sess.run([trained_h1, trained_h2, trained_hout, trained_b1, trained_b2, trained_bout],
                    feed_dict={})

    np.save('data/weights/h1.npy', trained_h1)
    np.save('data/weights/h2.npy', trained_h2)
    np.save('data/weights/hout.npy', trained_hout)
    np.save('data/weights/b1.npy', trained_b1)
    np.save('data/weights/b2.npy', trained_b2)
    np.save('data/weights/bout.npy', trained_bout)
