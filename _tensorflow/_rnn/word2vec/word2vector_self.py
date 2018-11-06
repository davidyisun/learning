from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: word2vector 自己版
Created on 2018-11-06
@author:David Yisun
@group:data
"""
import numpy as np
import tensorflow as tf
import math
import collections
import random
from optparse import OptionParser
import os
import jieba
import codecs
import re


# 外部参数
def get_entry_config():
    usage = 'crawler_jinrongbaike_vocab_entry'
    parser = OptionParser(usage=usage)
    parser.add_option('--log_dir', action='store', dest='log_dir', type='str', default='../../../../model/word2vector/xiaoshuo/log/')
    parser.add_option('--data_dir', action='store', dest='data_dir', type='str', default='./data/280.txt')
    parser.add_option('--stop_words_dir', action='store', dest='stop_words_dir', type='str', default='./data/stop_words.txt')
    parser.add_option('--model_dir', action='store', dest='model_dir', type='str', default='../../../../data/model/word2vector/model/xiaoshuo/')

    option, args = parser.parse_args()
    res = {'log_dir': option.log_dir,
           'data_dir': option.data_dir,
           'stop_words_data': option.stop_words_dir,
           'model_dir': option.model_dir}
    return res

para = get_entry_config()
for key in para:
    if key.endswith('_dir'):
        if not os.path.exists(para[key]):
            os.makedirs(para[key])


# 词向量模型
class word2vec():
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.  # 负样本数

    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(a=valid_window, size=valid_size, replace=False, p=None)  # 在range(a)中以概率P随机选size个数，replace为是否又放回

    learn_rate = 1.0  # 学习率

    def __init__(self, vocabulary_size, dictionary, reverse_dictionary, log_dir, model_dir, data_dir):
        self.vocabulary_size = vocabulary_size
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.data_dir = data_dir

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        pass

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            with tf.name_scope('inputs'):
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                self.valid_dataset = tf.placeholder(tf.int32, shape=[None])  # 用于计算词相似度的数据集

            # covert word id to embedding
            with tf.name_scope('embeddings'):
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):   # 注意 此处 weight 的shape 为 [vocabulary_size, embedding_size]
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.nce_weights,
                        biases=self.nce_biases,
                        labels=self.train_labels,
                        inputs=self.embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary. 将loss存到tensorflow里面
            tf.summary.scalar('loss', self.loss)

            # Construct the SGD optimizer using a learning rate of 1.0. # 构造优化器
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings. # 正则化embedding 计算词与词之间的相似度
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / self.norm
            self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset) # 找出验证次的embedding并计算它们和所有单词的相似度

            self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)  # 计算相似度

            # Merge all summaries.
            self.merged = tf.summary.merge_all()

            # Add variable initializer. 初始化变量
            self.init = tf.global_variables_initializer()

            # Create a saver
            self.saver = tf.train.Saver()

    def generate_batch(self, data, data_index, batch_size, num_skips, skip_window):
        """
        :param data: 整个数据集
        :param data_index:
        :param batch_size:
        :param num_skips: 上下文词数
        :param skip_window: 窗口大小--中心词半径--中心词在一个window里面的位置
        :return:
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 被随机初始化
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]  整个词组的跨度
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin  设置buffer
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)  # 打乱上下文顺序
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels, data_index

    def train_by_sentence(self, data_id_list, num_steps=100001):
        self.data = data_id_list
        self.num_steps = num_steps  # 设置训练steps
        # Open a writer to write summaries. 将graph写进tensorboard
        writer = tf.summary.FileWriter(logdir=self.log_dir,
                                       graph=self.graph)
        average_loss = 0
        data_index = 0
        embeddings = []
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = self.generate_batch(data=self.data,
                                                                         data_index=data_index,
                                                                         batch_size=self.batch_size,
                                                                         num_skips=self.num_skips,
                                                                         skip_window=self.skip_window)
            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            # Define metadata variable. 定义元数据变量 )定义Tensorflow运行的元信息， 记录训练时运算时间和内存占用等方面的信息.
            run_metadata = tf.RunMetadata()
            _, summary, loss_val, embeddings = self.sess.run([self.optimizer, self.merged, self.loss, self.normalized_embeddings], feed_dict=feed_dict)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.记录最后一次运行的元信息
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:    # 每2000次重新算一下loss 和 average loss
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
                # Save the model for checkpoints  保存模型
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

            # Note that this is expensive (~20% slowdown if computed every 500 steps) 计算矩阵相似性
            if step % 10000 == 0:
                test_words, near_words, near_sims, sim_mean, sim_var = self.cal_similarity(word_id_list=self.valid_examples.tolist(), top_k=8)
                for i in range(len(test_words)):
                    s = 'Nearest to {0} is {1}: {2}, sim is {3} \n'.format(test_words[i], i+1, near_words[i], near_sims[i])
                    print(s)
        self.final_embeddings = embeddings
        return

    def cal_similarity(self, word_id_list, top_k=8):
        sim_matrix = self.sess.run(self.similarity, feed_dict={self.valid_dataset: word_id_list})
        sim_mean = np.mean(sim_matrix)
        sim_var = np.mean(np.square(sim_matrix-sim_mean))
        test_words = []
        near_words = []
        near_sims = []
        for i in range(len(word_id_list)):
            test_words.append(self.reverse_dictionary[word_id_list[i]])
            nearest_id = (-sim_matrix[i,:]).argsort()[1:top_k+1]
            nearest_sim = -sim_matrix[i, nearest_id]
            near_sims.append(nearest_sim)
            nearest_word = [self.reverse_dictionary[x] for x in nearest_id]
            near_words.append(nearest_word)
        return test_words, near_words, near_sims, sim_mean, sim_var


# 生成词表
def build_dataset():
    pass


# 文件预预处理: 获取文件, 去掉停用词, 生成词表
def data_preprocess():
    # 1.读取停用词 文件需要为一行一词
    stop_words = []
    with codecs.open(para['stop_words_dir'], 'r', 'utf-8') as f:
        data = f.readlines()
    stop_words = set(data)

    # 2.读取文本，过滤停用词
    data_input = []
    with codecs.open(para['data_dir'], 'r', 'gbk') as f:
        data = f.read()
    data = re.sub(' |\\n|\\r', '', data)
    data_list = data.splitlines()
    data_list = jieba.lcut(data_list, cut_all=False)
    for word in data_list:
        if word not in stop_words:
            data_input.append(word)

    # 3.统计词频,选前30000个词
    word_count = collections.Counter(data_input)
    print('文本中总共有{n1}个单词,不重复单词数{n2}'.format(n1=len(data_input), n2=len(word_count)))
    word_count = word_count.most_common(50000)
    word_count = dict(word_count)  # word:count

    # 4.生成词表
    vocabulary = list(word_count.keys())
    dictionary = {'UNK':0}
    reverse_dictionary = {0:'UNK'}
    for index, w in enumerate(vocabulary):
        dictionary[w] = index   # w : id
        reverse_dictionary[index+1] = w   # id : w

    # 5.将原文换成id
    data_id_list = [dictionary[w] for w in data_input]

    return data_id_list, word_count, dictionary, reverse_dictionary


def train_main():
    # step 1 数据预处理
    data_id_list, word_count, dictionary, reverse_dictionary = data_preprocess()
    # step 2 建立模型并初始化
    model = word2vec(vocabulary_size=len(word_count),
                     dictionary=dictionary,
                     reverse_dictionary=reverse_dictionary,
                     log_dir=para['log_dir'],
                     model_dir=para['model_dir'],
                     data_dir=para['data_dir'])
    model.build_graph()
    model.init_op()
    # step 3 开始训练
    model.train_by_sentence(data_id_list=data_id_list)
    return

def inference_main():
    pass

if __name__ == '__main__':
    data_preprocess()