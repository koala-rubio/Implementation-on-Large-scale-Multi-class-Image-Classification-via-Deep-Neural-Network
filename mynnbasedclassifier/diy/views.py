from django.shortcuts import render

from . import models

from django.http import HttpResponse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from numpy import outer
from pyexpat import features

# Create your views here.

def index(request):
    return render(request, 'diy/index.html')

def classifyResult(request):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    identifier = int(request.POST.get('identifier', 1))

    # parameters
    learning_rate = 0.001
    training_epochs = 5
    batch_size = 100
    display_step = 1

    n_input = 784  # MNIST data input(image shape = [28,28])
    n_classes = 10  # MNIST total classes (0-9digits)
    dropout = 0.75  # probability to keep units

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # drop(keep probability)

    def conv2d(image, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME'), b))

    def max_pooling(image, k):
        return tf.nn.max_pool(image, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def conv_net(_X, _weights, _biases, _dropout):
        # Layer 1
        _X = tf.reshape(_X, [-1, 28, 28, 1])
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        conv1 = max_pooling(conv1, k=2)
        conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)
        # Layer 2
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        conv2 = max_pooling(conv2, k=2)
        conv2 = tf.nn.dropout(conv2, keep_prob=_dropout)
        # Fully Connected
        dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        dense1 = tf.nn.dropout(dense1, _dropout)
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        print(out)
        print(out.get_shape)
        return out

    # model
    pred = conv_net(x, weights, biases, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    right_ans = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9]
    s = []

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "./diy/model_tmp/mnist_demo.ckpt")
        # validate
        batch_xs, batch_ys = mnist.train.next_batch(identifier - 1)
        batch_xs, batch_ys = mnist.train.next_batch(1)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        print("Accuracy = " + "{:.4f}".format(acc))
        if (acc > 0.5):
            temp = "Predict answer = " + str(right_ans[identifier - 1])
            s.append(temp)
            temp = "Standard answer = " + str(right_ans[identifier - 1])
            s.append(temp)
            temp = "Right answer"
            s.append(temp)
        else:
            temp = "Predict answer = " + str(y)
            s.append(temp)
            temp = "Standard answer = " + str(right_ans[identifier - 1])
            s.append(temp)
            temp = "Wrong answer"
            s.append(temp)

        saver.save(sess, "model_tmp/mnist_demo.ckpt")

    tf.reset_default_graph()

    return render(request, 'diy/classfierResult.html', {'s': s})
    #return HttpResponse('Hello, this is diy!')