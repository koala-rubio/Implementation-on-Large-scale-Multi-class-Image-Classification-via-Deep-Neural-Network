from django.shortcuts import render

from . import models

from django.http import HttpResponse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from numpy import outer
from pyexpat import features

# Create your views here.

def index(request):
    return render(request, 'mnist/index.html')

def typeChoose(request):
    nn_type = request.POST.get('nn_type', 'BP')
    return render(request, 'mnist/para.html', {'nn_type': nn_type})
    #return HttpResponse('Welcome, this method is '+nn_type+'.')

def hitRate(request):
    nn_type = request.POST.get('nn_type', 'BP')
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # parameters
    if nn_type == 'BP':
        learning_rate = float(request.POST.get('learning_rate', 0.5))
    else:
        learning_rate = float(request.POST.get('learning_rate', 0.001))

    training_epochs = int(request.POST.get('training_epochs', 5))
    batch_size = int(request.POST.get('batch_size', 100))
    display_step = int(request.POST.get('display_step', 1))

    # initial output string
    s = []
    temp = "-------------------Parameters setting-------------------"
    s.append(temp)
    temp = "Method : " + nn_type
    s.append(temp)
    temp = "Learning_rate = " + "{:.4f}".format(learning_rate)
    s.append(temp)
    temp = "Training_epochs = " + '%04d' % training_epochs
    s.append(temp)
    temp = "Batch_size = " + '%04d' % batch_size
    s.append(temp)
    temp = "Display_step = " + '%04d' % display_step
    s.append(temp)
    temp = "-------------------Training progress-------------------"
    s.append(temp)

    if nn_type == 'BP':

        # initialization
        x = tf.placeholder(tf.float32, [None, 784])  # ? * 784 (28^2)
        y_ = tf.placeholder(tf.float32, [None, 10])  # ? * 10 types
        W = tf.Variable(tf.zeros([784, 10]))  # weight
        b = tf.Variable(tf.zeros([10]))  # bias counter
        y = tf.nn.softmax(tf.matmul(x, W) + b)  # prediction x matrix(*) W matrix(+) b

        # model
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.log(y)))  # cross entropy
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  # minimize loss function
        init = tf.global_variables_initializer()  # initialize model
        sess = tf.InteractiveSession()  # build interactive session
        sess.run(init)  # run

        # train
        for epoch in range(training_epochs):
            total_batch = int(mnist.train._num_examples / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
            if epoch % display_step == 0:
                correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # compare
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # transform type
                temp = "Epoch : " + '%04d' % (epoch + 1) + ", Accuracy = " + "{:.4f}".format(
                    sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) + "\n"
                s.append(temp)
            # print(s)
            # print("Epoch:", '%04d' % (epoch+1), "Accuracy =", "{:.4f}".format(sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels})))

        temp = "-------------------Final result-------------------"
        s.append(temp)
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # compare
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # transform type
        temp = "Epoch : " + '%04d' % (epoch + 1) + ", Accuracy = " + "{:.4f}".format(
            sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # test and output
        s.append(temp)

        #return HttpResponse(nn_type+' '+movement)

    elif nn_type == 'CNN':

        # network Parameters
        n_input = 784  # MNIST data input(image shape = [28,28])
        n_classes = 10  # MNIST total classes (0-9digits)
        dropout = 0.75  # probability to keep units

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)  # drop(keep probability)

        # step definition
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
            return out

        # model
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, "model_tmp/mnist_demo.ckpt")
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(mnist.train._num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if epoch % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                        loss) + ", Accuracy = " + "{:.4f}".format(acc)
                    s.append(temp)
            saver.save(sess, "model_tmp/mnist_demo.ckpt")

            temp = "-------------------Final result-------------------"
            s.append(temp)
            temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                loss) + ", Accuracy = " + "{:.4f}".format(acc)
            s.append(temp)

    elif nn_type == 'CNN_SIGMOID':

        # network Parameters
        n_input = 784  # MNIST data input(image shape = [28,28])
        n_classes = 10  # MNIST total classes (0-9digits)
        dropout = 0.75  # probability to keep units

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)  # drop(keep probability)

        # step definition
        def conv2d(image, w, b):
            return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME'), b))

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
            dense1 = tf.nn.sigmoid(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
            dense1 = tf.nn.dropout(dense1, _dropout)
            out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
            print(out)
            return out

        # model
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(mnist.train._num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if epoch % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                        loss) + ", Accuracy = " + "{:.4f}".format(acc)
                    s.append(temp)

            temp = "-------------------Final result-------------------"
            s.append(temp)
            temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                loss) + ", Accuracy = " + "{:.4f}".format(acc)
            s.append(temp)

    elif nn_type == 'CNN_DROPOUT':

        # network Parameters
        n_input = 784  # MNIST data input(image shape = [28,28])
        n_classes = 10  # MNIST total classes (0-9digits)
        dropout = 0.75  # probability to keep units

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)  # drop(keep probability)

        # step definition
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
            # conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)
            # Layer 2
            conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
            conv2 = max_pooling(conv2, k=2)
            # conv2 = tf.nn.dropout(conv2,keep_prob=_dropout)
            # Fully Connected
            dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
            dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
            # dense1 = tf.nn.dropout(dense1,_dropout)
            out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
            print(out)
            return out

        # model
        pred = conv_net(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(mnist.train._num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if epoch % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                        loss) + ", Accuracy = " + "{:.4f}".format(acc)
                    s.append(temp)

            temp = "-------------------Final result-------------------"
            s.append(temp)
            temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                loss) + ", Accuracy = " + "{:.4f}".format(acc)
            s.append(temp)

    elif nn_type == 'CNN_ARCH':

        prediction = 0.1

        # step definition
        def compute_accuracy(v_xs, v_ys):
            y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
            return result

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            # stride [1, x_movement, y_movement, 1]
            # Must have strides[0] = strides[3] = 1
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            # stride [1, x_movement, y_movement, 1]
            # ksize  [1,pool_op_length,pool_op_width,1]
            # Must have ksize[0] = ksize[3] = 1
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # define placeholder for inputs to network
        xs = tf.placeholder(tf.float32, [None, 784])  # ? * 784 (28^2)
        ys = tf.placeholder(tf.float32, [None, 10])  # ? * 10 types
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1, 28, 28, 1])

        ## conv1 layer ##
        W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
        h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

        ## conv2 layer ##
        W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
        h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

        ##flat h_pool2##
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

        ## fc1 layer ##
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        ## fc2 layer ##
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        # model
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # train
        for epoch in range(training_epochs):
            total_batch = int(mnist.train._num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
            if epoch % display_step == 0:
                temp = "Epoch : " + '%04d' % (epoch + 1) + ", Accuracy = " + "{:.4f}".format(
                    compute_accuracy(mnist.test.images, mnist.test.labels))
                s.append(temp)

        temp = "-------------------Final result-------------------"
        s.append(temp)
        temp = "Epoch : " + '%04d' % (epoch + 1) + ", Accuracy = " + "{:.4f}".format(
            compute_accuracy(mnist.test.images, mnist.test.labels))
        s.append(temp)

    elif nn_type == 'DNN':

        # network parameters
        n_input = 784
        n_classes = 10
        dropout = 0.8

        # tf Graph input
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)

        # step definition
        def conv2d(image, w, b):
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME'), b))

        def max_pool(image, k):
            return tf.nn.max_pool(image, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        def dnn(_X, _weights, _biases, _dropout):
            _X = tf.nn.dropout(_X, _dropout)
            d1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(_X, _weights['wd1']), _biases['bd1']), name='d1')
            d2x = tf.nn.dropout(d1, _dropout)
            d2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d2x, _weights['wd2']), _biases['bd2']), name='d2')
            dout = tf.nn.dropout(d2, _dropout)
            out = tf.matmul(dout, weights['out']) + _biases['out']
            return out

        weights = {
            'wd1': tf.Variable(tf.random_normal([784, 600], stddev=0.01)),
            'wd2': tf.Variable(tf.random_normal([600, 480], stddev=0.01)),
            'out': tf.Variable(tf.random_normal([480, 10]))
        }

        biases = {
            'bd1': tf.Variable(tf.random_normal([600])),
            'bd2': tf.Variable(tf.random_normal([480])),
            'out': tf.Variable(tf.random_normal([10])),
        }

        # model
        pred = dnn(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                total_batch = int(mnist.train._num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if epoch % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                        loss) + ", Training Accuracy = " + "{:.4f}".format(acc)
                    s.append(temp)

            temp = "-------------------Final result-------------------"
            s.append(temp)
            temp = "Epoch : " + '%04d' % (epoch + 1) + ", Minibatch Loss = " + "{:.4f}".format(
                loss) + ", Training Accuracy = " + "{:.4f}".format(acc)
            s.append(temp)

    else:

        # network parameters
        n_hidden_1 = 256  # 1st layer num features
        n_hidden_2 = 256  # 2nd layer num features
        n_input = 784
        n_classses = 10

        # define placeholder for inputs to network
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classses])

        # step definition
        def multilayer_perceptron(_X, _weights, _biases):
            layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
            layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2']))
            return tf.matmul(layer2, _weights['out']) + _biases['out']

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classses]))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classses]))
        }

        # model
        pred = multilayer_perceptron(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train._num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                    avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                if epoch % display_step == 0:
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    temp = "Epoch : " + '%04d' % (epoch + 1) + ", Cost = " + "{:.4f}".format(
                        avg_cost) + ", Accuracy = " + "{:.4f}".format(
                        accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
                    s.append(temp)

            temp = "-------------------Final result-------------------"
            s.append(temp)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            temp = "Epoch : " + '%04d' % (epoch + 1) + ", Cost = " + "{:.4f}".format(
                avg_cost) + ", Accuracy = " + "{:.4f}".format(
                accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            s.append(temp)
            # print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    models.training_record_mnist.objects.create(record_mnist=s)

    return render(request, 'mnist/accuracy.html', {'s': s})