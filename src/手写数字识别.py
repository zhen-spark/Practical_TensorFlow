# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:04:35 2020

@author: Administrator
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
my_mnist = input_data.read_data_sets("C:/Users/Administrator/MNIST_data_bak/", one_hot=True)

# The MNIST data is split into three parts:
# 55,000 data points of training data (mnist.train)
# 10,000 points of test data (mnist.test), and
# 5,000 points of validation data (mnist.validation).

# Each image is 28 pixels by 28 pixels

# 输入的是一堆图片，None表示不限输入条数，784表示每张图片都是一个784个像素值的一维向量
# 所以输入的矩阵是None乘以784二维矩阵
x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练
# labels是每张图片都对应一个one-hot的10个值的向量
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))
# 定义损失函数，交叉熵损失函数
# 对于多分类问题，通常使用交叉熵损失函数
# reduction_indices等价于axis，指明按照每行加，还是按照每列加
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 评估

# tf.argmax()是一个从tensor中寻找最大值的序号，tf.argmax就是求各个预测的数字中概率最大的那一个

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 创建Saver节点，用于保存训练的模型
saver = tf.train.Saver()
for i in range(100):
    batch_xs, batch_ys = my_mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 每隔一段时间保存一次中间结果
    if i % 10 == 0:
        save_path = saver.save(sess, "C:/Users/Administrator/MNIST_data_bak/saver/softmax_middle_model.ckpt")

# 测试
print("TestSet acc : %s" % accuracy.eval({x: my_mnist.test.images, y_: my_mnist.test.labels}))
# 保存最终的模型
save_path = saver.save(sess, "C:/Users/Administrator/MNIST_data_bak/saver/softmax_final_model.ckpt")

# 使用训练好的模型直接进行预测
with tf.Session() as sess_back:
    saver.restore(sess_back, "C:/Users/Administrator/MNIST_data_bak/saver/softmax_final_model.ckpt")
    # 评估
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accruary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 测试
    print(accuracy.eval({x : my_mnist.test.images, y_ : my_mnist.test.labels}))