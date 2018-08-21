import numpy as np
import tensorflow as tf
import argparse
from helper import *

model_name = '3dense'
image_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def forward_propagation(X):

    dense1 = tf.layers.dense(inputs=X, units=500, kernel_initializer=tf.random_normal_initializer,
                             bias_initializer=tf.zeros_initializer(),
                             activation=tf.nn.relu, name='dense1')

    dense2 = tf.layers.dense(inputs=dense1, units=784, kernel_initializer=tf.random_normal_initializer,
                             bias_initializer=tf.zeros_initializer(),
                             activation=tf.nn.relu,
                             name='dense2')

    dense3 = tf.layers.dense(inputs=dense2, units=10, kernel_initializer=tf.random_normal_initializer,
                             bias_initializer=tf.zeros_initializer(),
                             #activation=tf.nn.relu,
                             name='dense3')

    '''
    dense4 = tf.layers.dense(inputs=dense3, units=10, kernel_initializer=tf.random_normal_initializer,
                             bias_initializer=tf.zeros_initializer(),
                             name='dense4')
    '''
    return dense3

def compute_cost(scores, y):
    logits = scores
    labels = y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=labels))
    return cost

def get_minibatch(X, y, mb_size):
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm][:mb_size], one_hot(y[perm][:mb_size], 10, 1)
    '''

    m = X.shape[0]
    mini_batches = []

    # Shuffling the data
    permutation = list(np.random.permutation(m))
    shuf_x = X[permutation, :]
    shuf_y = y[permutation,]

    # Partition
    total_mb = np.floor(m / mb_size).astype(np.int32)

    for k in range(total_mb):
        mb_x = shuf_x[k * mb_size : (k+1) * mb_size, :]
        mb_y = shuf_y[k * mb_size : (k+1) * mb_size,]
        mini_batches.append((mb_x, mb_y))

    if m % mb_size != 0:
        mb_x = shuf_x[(total_mb * mb_size) :, :]
        mb_y = shuf_y[(total_mb * mb_size) :,]
        mini_batches.append((mb_x, mb_y))

    return mini_batches

def save_graph_to_file(sess, graph, graph_file_name='model.pb'):
    output_graph_def = graph_util.convert_variables_to_constants(sess,
                                    graph.as_graph_def(), [FLAGS.final_])
def model(X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=1500,
            mb_size=200):

    y_train_oh = one_hot(y_train, 10, 1)
    y_val_oh = one_hot(y_val, 10, 1)
    X_val = reshape_data(X_val)

    m, n_x = X_train.shape
    n_y = y_train_oh.shape[1]

    X, y = create_placeholder_variables(n_x, n_y)

    scores = forward_propagation(X)

    cost = compute_cost(scores, y)
    sum_cost = tf.summary.scalar('cost', cost)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    prediction = tf.argmax(scores, 1, name='prediction')

    acc_vec = tf.cast(tf.equal(tf.cast(y_val, tf.int64), prediction), dtype=tf.int32,
                      name='acc_vec1')
    acc = tf.divide(tf.reduce_sum(acc_vec), tf.shape(X)[0], name='accuracy1')
    accuracy = tf.summary.scalar('accuracy', acc)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs/' + model_name + '/')
        writer.add_graph(sess.graph)

        sess.run(init)

        step = 1
        for i in range(epochs):
            mini_batches = get_minibatch(X_train, y_train, mb_size)

            j = 1

            for mb in mini_batches:
                mb_x, mb_y = mb

                mb_y = one_hot(mb_y, 10, 1)

                feed_dict = {X : mb_x,
                             y : mb_y}

                _, n_cost = sess.run([optimizer, cost], feed_dict=feed_dict)

                print (n_cost, j)
                j += 1
                step += 1

                if j % 10 == 0:
                    s = sess.run(merged, feed_dict={X:X_val, y:y_val_oh})
                    writer.add_summary(s, step)


        '''
        save_path = saver.save(sess, './var_dir/model.ckpt')
        print ('Model saved in path: %s' % save_path)
        '''
        writer.close()

    return

def run_model_with_data():
    path = './mnist.npz'
    X_train, y_train, X_test, y_test = load_data(path)

    # Just first 5000 examples for faster training
    '''
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    X_val = X_test[:100]
    y_val = X_test[100:200]
    '''
    X_val = X_test[:5000]
    y_val = y_test[:5000]

    X_test = X_test[5000:]
    y_test = y_test[5000:]

    X_train = reshape_data(X_train)
    model(X_train, y_train, X_val, y_val, mb_size=256, epochs=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
        type=str,
        default='mnist_neural_net',
        help='Name of the model you are building. Note: This name will be used\
              to create directory when visualizing model summary with Tensorboard.')

    parser.add_argument('--output_graph',
        type=str,
        default='./output_graph.pb',
        help='Where to save the model')

    parser.add_argument('--output_labels',
        type=str,
        default='output_labels.txt',
        help='Where to save the labels for the graph')

    parser.add_argument('--tb_dir',
        type=str,
        default='./tf_summaries/',
        help='Where to save the summaries for TensorBoard')

    parser.add_argument('--epochs',
        type=int,
        default=500,
        help='How many epochs to run')

    parser.add_argument('--learning_rate',
        type=float,
        default=0.01,
        help='The learning rate to use while training the model')

    parser.add_argument('--data_dir',
        type=str,
        default='./mnist/',
        help='The folder where the MNIST data is located. Do note the program \
            assumes that there is a file "mnist.npz" in the directory mentioned with\
             this argument.\n The required dataset can be downloaded from here: \
             https://www.kaggle.com/vikramtiwari/mnist-numpy')
             
    run_model_with_data()
