#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: seq2seq test
Created on 2018-09-25
@author:David Yisun
@group:data
"""


def t1():
    import tensorflow as tf
    import numpy as np

    n_steps = 2
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

    seq_length = tf.placeholder(tf.int32, [None])
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                        sequence_length=seq_length)

    init = tf.global_variables_initializer()

    X_batch = np.array([
        # step 0     step 1
        [[0, 1, 2], [9, 8, 7]],  # instance 1
        [[3, 4, 5], [0, 0, 0]],  # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]],  # instance 3
        [[9, 0, 1], [3, 2, 1]],  # instance 4
    ])
    seq_length_batch = np.array([2, 1, 2, 2])

    with tf.Session() as sess:
        init.run()
        outputs_val, states_val = sess.run(
            [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
        # outputs_shape = (outputs_val.state_size(), outputs_val.output_size())
        print("outputs_val.shape:", type(outputs_val), "states_val.shape:", type(states_val))
        print("outputs_val:", outputs_val)
        print("states_val:", states_val)

def t2():
    import tensorflow as tf
    import numpy as np
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))

def t3():
    # import tensorflow.contrib.eager as tfe
    import tensorflow as tf
    # tfe.enable_eager_execution()
    import numpy as np

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "b": np.random.uniform(size=(5, 2))
        }
    )
    a = dataset.make_one_shot_iterator()
    b = a.get_next()
    s = tf.Session()
    for i in range(5):
        print(s.run(b))
        # print(a)


if __name__ == '__main__':
    t3()

