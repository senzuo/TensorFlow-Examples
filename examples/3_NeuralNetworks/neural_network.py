""" Neural Network.
Use TensorFlow '*layers*' and '*estimator*' API to build a simple neural network
(a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

问题
    1. 各个参数所代表的意义?
        一般参数
            learning_rate   梯度下降法的参数
            batch_size      每次迭代所需要的样本数量
            num_epochs      一个epoch是指把所有训练数据完整的过一遍
            num_steps       ??
        网络参数
            n_hidden_1      第一层hidden layer的神经元的个数

    2. model_fn函数的参数是默认的吗?
        feature:    可能和输入的通道和batch_size有关
        label:      ??
        mode:       可能和model的调用方法有关

    3. 测试阶段model_fn为什么还要执行一遍?
            model定义里面已经有预测值了吗?
            ??
    4. 为什么要加from __future__ import print_function

注意
    logits: 未归一化的概率， 一般也就是 softmax的输入
总结
    跟随网络查看每个batch的大小
        [batch_size × 784] * [784 * n_hidden_1] -> [batch_size × n_hidden_1]
        [batch_size × n_hidden_1] * [n_hidden_1 * n_hidden_2] -> [batch_size × n_hidden_2]
        [batch_size × n_hidden_2] * [n_hidden_2 * output10] -> [batch_size × output10]

"""
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    # x(input).shape = batch_size × 784
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    # layer_1(output of layer1).shape = batch_size × n_hidden_1
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    # layer_2(output of layer2).shape = batch_size × n_hidden_2
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    # out_layer.shape = batch_size × 10
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # 这里的labels同为batch_size × 1
    # Build the neural network
    # 未归一化的概率
    logits = neural_net(features)

    # Predictions
    # logits沿着1轴,取最大可能性的输出结果就为预测的类
    pred_classes = tf.argmax(logits, axis=1)
    # softmax回归,得到10个概率
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer 损失函数和优化方法
    # reduce_mean:求平均值
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)
        )
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
# input_fn is a function
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False
)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
