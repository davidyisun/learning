from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#!/usr/bin/env python
"""
    脚本名: w2v basic 版本
Created on 2018-11-02
@author:David Yisun
@group:data
"""
"""Basic word2vec example."""



import collections
import math
import os
import sys
import argparse
import random
from sklearn.manifold import TSNE
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector


# --- 系统参数 ---
# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.   用于储存tensorboard的日志文件放在当前文件夹下
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))   # os.path.dirname 获取父级路径

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default='../../../../model/word2vector/log',
    help='The log directory for TensorBoard summaries.'
)
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# --- 第一步：下载数据 --- Step1: Download the data
url = 'http://mattmahoney.net/dc/'

# pylint: disable = redefined-outer-name
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = '../../../../data/learning/word2vector/'+filename    # tempfile 操作临时文件夹  gettempdir()为临时文件夹的路径
    if not os.path.exists(local_filename):
        print('down the data ......')
        local_filename, _ = urllib.request.urlretrieve(url + filename,   # urllib.request.urlretrieve 为下载文件
                                                       local_filename)
    statinfo = os.stat(local_filename)  # 获取文件信息 如 权限 user id 文件大小等
    if statinfo.st_size == expected_bytes:  # 判断文件大小是否为指定大小 从而判断 是否 为所需文件
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename  #


# Read the data into a list of strings. 读取数据
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:                     # 解压文件
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data


# --- 第二步：生成词表 --- Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]  # count为词频list
  count.extend(collections.Counter(words).most_common(n_words - 1))  # 取词频在由高到低前n_words个词入表
  dictionary = dict()    # 'UNK' id为 0
  for word, _ in count:
    dictionary[word] = len(dictionary)   # dictionary: 单词:id
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # reversed_dictionary: id:单词
  return data, count, dictionary, reversed_dictionary


# --- 第三步：生成batch --- Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, data_index, batch_size, num_skips, skip_window):
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
    words_to_use = random.sample(context_words, num_skips)   # 打乱上下文顺序
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


# --- 第四步：模型建立 --- Build and train a skip-gram model.
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


    def __init__(self, vocabulary_size, data, dictionary, reverse_dictionary):
        self.vocabulary_size = vocabulary_size
        self.data = data
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary

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

    def train_by_sentence(self, num_steps=100001):
        self.num_steps = num_steps  # 设置训练steps
        # Open a writer to write summaries. 将graph写进tensorboard
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir,
                                       graph=self.graph)
        average_loss = 0
        data_index = 0
        embeddings = []
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = generate_batch(data=self.data,
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
                self.saver.save(self.sess, os.path.join(FLAGS.log_dir, 'model.ckpt'), global_step=step)

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


# --- 第五步：开始训练 ---  Step 5: Begin training.
def plot_with_labels(low_dim_embs, labels, filename):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

def main():
    # --- 第一步：下载数据 --- Step1: Download the data
    # filename = maybe_download('text8.zip', 31344016)
    # vocabulary = read_data(filename)
    s = '../../../../data/learning/word2vector/text8.zip'
    vocabulary = read_data(s)
    print('Data size', len(vocabulary))

    # --- 第二步：生成词表 --- Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size) # data:id 序列 count：词频
    del vocabulary  # Hint to reduce memory.
    # print('Most common words (+UNK)', count[:5])
    # print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # --- 第三步：生成batch --- Step 3: Function to generate a training batch for the skip-gram model.

    # batch, labels, data_index = generate_batch(data=data, data_index=data_index, batch_size=8, num_skips=2, skip_window=1)
    # for i in range(8):
    #     print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
    #           reverse_dictionary[labels[i, 0]])

    # # --- 第四步：模型建立 --- Build and train a skip-gram model.
    model = word2vec(vocabulary_size=vocabulary_size,
                     data=data,
                     dictionary=dictionary,
                     reverse_dictionary=reverse_dictionary)
    model.build_graph()
    print('Build the model and graph')

    # --- 第五步：开始训练 ---  Step 5: Begin training.
    model.init_op()  # 初始化
    print('Initialized')
    model.train_by_sentence()  # 训练图
    # --- 第六步：模型可视化 --- Step 6: Visualize the embeddings.

    # 因为我们的embedding的大小为128维，没有办法直接可视化，所以我们用t-SNE方法进行降维
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # 只画出500个词的位置
    plot_only = 500
    if model.final_embeddings != []:
        low_dim_embs = tsne.fit_transform(model.final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, '../../../../model/word2vector/tsne.png')
    pass


if __name__ == '__main__':
    main()
