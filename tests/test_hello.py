# TensorFlow入门：第一个机器学习Demo
#   https://blog.csdn.net/geyunfei_/article/details/78782804

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sess = tf.Session()
# ============================================= #
# t0 = tf.constant(3, dtype=tf.int32)
# t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)
# t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)
# t3 = tf.constant([[[5],[6],[7]], [[4],[3],[2]]])

# print(sess.run(t0))
# print(sess.run(t1))
# print(sess.run(t2))
# print(sess.run(t3))

# ============================================= #
# node1 = tf.constant(3.2)
# node2 = tf.constant(4.8)
# adder = node1 + node2
# print(adder)
# print(sess.run(adder))

## ============================================= ##
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder = a + b
# muler = a * b
# print(sess.run(adder, {a:3, b:4.2}))
# print(sess.run(adder, {a:[1,2], b:[3,4]}))
# print(sess.run(muler, {a:3, b:4.5}))

## ============================================= ##
# w = tf.Variable([0.1], dtype=tf.float32)
# b = tf.Variable([-0.1], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = w * x + b
# y = tf.placeholder(tf.float32)
# loss = tf.reduce_sum(tf.square(linear_model - y))

# init = tf.global_variables_initializer()
# sess.run(init)

# way 1
# print(sess.run(linear_model, {x:[1,2,3,6,8]}))
# print(sess.run(loss, {x:[1,2,3,6,8], y:[4.8, 8.5, 10.4, 21.0, 25.3]}))

# w_fix = tf.assign(w, [2.])
# b_fix = tf.assign(b, [1.])

# sess.run([w_fix, b_fix])

# print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# way 2
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train = optimizer.minimize(loss)

# x_train = [1,2,3,6,8]
# y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

# for i in range(1000):
#     sess.run(train, {x: x_train, y: y_train})

# print('w: {} | b: {} | loss: {}'.format(sess.run(w), sess.run(b), sess.run(loss, {x:x_train, y:y_train})))

## ============================================= ##

# feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]
# estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# x_train = np.array([1., 2., 3., 6., 8.])
# y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

# x_eval = np.array([2., 5., 7., 9.])
# y_eval = np.array([7.6, 17.2, 23.6, 28.8])

# # 用训练数据创建一个输入模型，用来进行后面的模型训练
# # 第一个参数用来作为线性回归模型的输入数据
# # 第二个参数用来作为线性回归模型损失模型的输入
# # 第三个参数batch_size表示每批训练数据的个数
# # 第四个参数num_epochs为epoch的次数，将训练集的所有数据都训练一遍为1次epoch
# # 第五个参数shuffle为取训练数据是顺序取还是随机取
# train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)
# # 再用训练数据创建一个输入模型，用来进行后面的模型评估
# train_input_fn_2 = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)
# # 用评估数据创建一个输入模型，用来进行后面的模型评估
# eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=2, num_epochs=1000, shuffle=False)
# # 使用训练数据训练1000次
# estimator.train(input_fn=train_input_fn, steps=1000)
# # 使用原来训练数据评估一下模型，目的是查看训练的结果
# train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
# print('train metrics: {}'.format(train_metrics))
# # 使用评估数据评估一下模型，目的是验证模型的泛化性能
# eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
# print('eval metrics: {}'.format(eval_metrics))

## ============================================= ##
# 定义模型训练函数，同时也定义了特征向量
def model_fn(features, labels, mode):
    # 构建线性模型
    w = tf.get_variable('w', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = w * features['x'] + b
    # 构建损失模型
    loss = tf.reduce_sum(tf.square(y - labels))
    # 训练模型子图
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # 通过 EstimatorSpec 指定我们的训练子图以及损失模型
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

# 创建自定义的训练模型
estimator = tf.estimator.Estimator(model_fn=model_fn)

# 后面的训练逻辑与使用 LinearRegressor 一样
x_train = np.array([1., 2., 3., 6., 8.])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])
x_eavl = np.array([2., 5., 7., 9.])
y_eavl = np.array([7.6, 17.2, 23.6, 28.8])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=None, shuffle=True)
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eavl}, y_eavl, batch_size=2, num_epochs=1000, shuffle=False)
estimator.train(input_fn=train_input_fn, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_fn_2)
print("train metrics: %r" % train_metrics)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %s" % eval_metrics)
