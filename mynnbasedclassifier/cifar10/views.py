from django.shortcuts import render

from . import models

from django.http import HttpResponse

import tensorflow as tf
import pickle
import os
import numpy as np
import math


class Cifar10DataReader():
    def __init__(self, cifar_folder, onehot=True):
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.data_index = 1
        self.read_next = True
        self.data_label_train = None
        self.data_label_test = None
        self.batch_index = 0

    def unpickle(self, f):
        fo = open(f, 'rb')
        d = pickle.load(fo, encoding="bytes")
        fo.close()
        return d

    def next_train_data(self, batch_size=100):
        assert 10000 % batch_size == 0, "10000%batch_size!=0"
        rdata = None
        rlabel = None
        if self.read_next:
            f = os.path.join(self.cifar_folder, "data_batch_%s" % (self.data_index))
            print('read:', f)

            dic_train = self.unpickle(f)
            self.data_label_train = list(zip(dic_train[b'data'], dic_train[b'labels']))  # label 0~9
            np.random.shuffle(self.data_label_train)

            self.read_next = False
            if self.data_index == 5:
                self.data_index = 1
            else:
                self.data_index += 1

        if self.batch_index < len(self.data_label_train) // batch_size:
            # print self.batch_index
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            rdata, rlabel = self._decode(datum, self.onehot)
        else:
            self.batch_index = 0
            self.read_next = True
            return self.next_train_data(batch_size=batch_size)

        return rdata, rlabel

    def _decode(self, datum, onehot):
        rdata = list()
        rlabel = list()
        if onehot:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                hot = np.zeros(10)
                hot[int(l)] = 1
                rlabel.append(hot)
        else:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                rlabel.append(int(l))
        return rdata, rlabel

    def next_test_data(self, batch_size=100):
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test_batch")
            print('read:', f)

            dic_test = self.unpickle(f)
            data = dic_test[b'data']
            labels = dic_test[b'labels']  # 0~9
            self.data_label_test = list(zip(data, labels))

        np.random.shuffle(self.data_label_test)
        datum = self.data_label_test[0:batch_size]

        return self._decode(datum, self.onehot)

# Create your views here.

def index(request):
    return render(request, 'cifar10/index.html')

def hitRate(request):

    training_num_each_epoch = int(request.POST.get('training_num_each_epoch', 50000))
    training_epochs = int(request.POST.get('training_epochs', 5))
    batch_size = int(request.POST.get('batch_size', 100))
    display_step = int(request.POST.get('display_step', 1))

    step = 0
    train_iter = training_num_each_epoch * training_epochs

    s = []
    temp = "-------------------Parameters setting-------------------"
    s.append(temp)
    temp = "Training_image_each_epoch = " + '%04d' % training_num_each_epoch
    s.append(temp)
    temp = "Training_epochs = " + '%04d' % training_epochs
    s.append(temp)
    temp = "Batch_size = " + '%04d' % batch_size
    s.append(temp)
    temp = "Display_step = " + '%04d' % display_step
    s.append(temp)
    temp = "-------------------Training progress-------------------"
    s.append(temp)

    # for key in data:
    #     print(key)
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    is_traing = tf.placeholder(tf.bool)

    ####conv1
    W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=5e-2))
    conv_1 = tf.nn.conv2d(input_x, W1, strides=(1, 1, 1, 1), padding="VALID")
    print(conv_1)

    bn1 = tf.layers.batch_normalization(conv_1, training=is_traing)

    relu_1 = tf.nn.relu(bn1)
    print(relu_1)

    pool_1 = tf.nn.max_pool(relu_1, strides=[1, 2, 2, 1], padding="VALID", ksize=[1, 3, 3, 1])
    print(pool_1)

    ####conv2
    W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=5e-2))
    conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="SAME")
    print(conv_2)

    bn2 = tf.layers.batch_normalization(conv_2, training=is_traing)

    relu_2 = tf.nn.relu(bn2)
    print(relu_2)

    pool_2 = tf.nn.max_pool(relu_2, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
    print(pool_2)

    ####conv3
    W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
    conv_3 = tf.nn.conv2d(pool_2, W3, strides=[1, 1, 1, 1], padding="SAME")
    print(conv_3)

    bn3 = tf.layers.batch_normalization(conv_3, training=is_traing)

    relu_3 = tf.nn.relu(bn3)
    print(relu_3)

    pool_3 = tf.nn.max_pool(relu_3, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")
    print(pool_3)

    # fc1
    dense_tmp = tf.reshape(pool_3, shape=[-1, 2 * 2 * 256])
    print(dense_tmp)

    fc1 = tf.Variable(tf.truncated_normal(shape=[2 * 2 * 256, 1024], stddev=0.04))

    bn_fc1 = tf.layers.batch_normalization(tf.matmul(dense_tmp, fc1), training=is_traing)

    dense1 = tf.nn.relu(bn_fc1)
    dropout1 = tf.nn.dropout(dense1, keep_prob)

    # fc2
    fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.04))
    out = tf.matmul(dropout1, fc2)
    print(out)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    dr = Cifar10DataReader(cifar_folder="./cifar10/cifar-10-batches-py/")

    # 测试网络
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 初始化所有的共享变量
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # 开启一个训练
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, "model_tmp/cifar10_demo.ckpt")
        step = 1
        display_round = training_num_each_epoch

        # Keep training until reach max iterations
        while step * batch_size < train_iter:
            step += 1
            batch_xs, batch_ys = dr.next_train_data(batch_size)
            # 获取批数据,计算精度, 损失值
            opt, acc, loss = sess.run([optimizer, accuracy, cost],
                                      feed_dict={input_x: batch_xs, y: batch_ys, keep_prob: 0.6, is_traing: True})
            if step % display_step == 0:
                print ("Epoch : " + '%04d' % int(step * batch_size / training_num_each_epoch) + ", Minibatch Loss = " + "{:.4f}".format(
                    loss) + ", Training Accuracy = " + "{:.4f}".format(acc))

            if (step * batch_size >= display_round):
                temp = "Epoch : " + '%04d' % int(step * batch_size / training_num_each_epoch) + ", Minibatch Loss = " + "{:.4f}".format(
                    loss) + ", Training Accuracy = " + "{:.4f}".format(acc)
                s.append(temp)
                display_round += training_num_each_epoch

        # 计算测试精度
        num_examples = 10000
        d, l = dr.next_test_data(num_examples)
        ans = sess.run(accuracy, feed_dict={input_x: d, y: l, keep_prob: 1.0, is_traing: True})
        print ("Testing Accuracy:", ans)
        temp = "-------------------Final result-------------------"
        s.append(temp)
        temp = "Epoch : " + '%04d' % training_epochs + ", Accuracy = " + "{:.4f}".format(ans)
        s.append(temp)
        saver.save(sess, "model_tmp/cifar10_demo.ckpt")
        models.training_record_cifar.objects.create(record_cifar=s)

        return render(request, 'cifar10/accuracy.html', {'s': s})