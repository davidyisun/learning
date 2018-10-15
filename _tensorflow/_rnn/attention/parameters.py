#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 参数设定 英文翻译成中文
Created on 2018-10-10
@author:David Yisun
@group:data
"""
# 参数设定 英文翻译成中文
# --- 训练参数  ---
class para_train(object):
    trg_train_data = './data/train.zh'   # 源语言输入文件。
    src_train_data = './data/train.en'    # 目标语言输入文件。
    checkpoint_path = '../../../../model/attention/en_to_zh/attention_ckpt'   # checkpoint保存路径。

    hidden_size = 1024  # LSTM的隐藏层规模。
    decoder_layers = 2  # 解码器中LSTM结构的层数。这个例子中编码器固定使用单层的双向LSTM。
    src_vocab_size = 10000  # 源语言词汇表大小。
    trg_vocab_size = 4000  # 目标语言词汇表大小。
    # --- 训练参数 ---
    batch_size = 100  # 训练数据batch的大小。
    num_epoch = 5  # 使用训练数据的轮数。
    keep_prob = 0.8  # 节点不被dropout的概率。
    max_grad_norm = 5  # 用于控制梯度膨胀的梯度大小上限。
    share_emb_and_softmax = True  # 在Softmax层和词向量层之间共享参数。

    max_len = 50  # 限定句子的最大单词数量。
    sos_id = 1  # 目标语言词汇表中<sos>的ID。


# --- 预测参数 ---
class para_predict(object):
    checkpoint_path = '../../../../model/attention/en_to_zh/attention_ckpt-9000'

    hidden_size = 1024
    decoder_layers = 2
    src_vocab_size = 10000
    trg_vocab_size = 4000
    share_emb_and_softmax = True

    trg_vocab = './data/zh.vocab'
    src_vocab = './data/en.vocab'

    # 词汇表中<sos>和<eos>的id。在解码过程中需要用<sos>作为第一步的输入，并检查是否是<eos>，因此需要知道这两个符号的id
    sos_id = 1
    eos_id = 2
    max_dec_len = 100  # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。

p_train = para_train()
p_predict = para_predict()


# 参数设定 中文翻译成英文
# --- 训练参数  ---
class para_train2(object):
    src_train_data = './data/train.zh'   # 源语言输入文件。
    trg_train_data = './data/train.en'    # 目标语言输入文件。
    checkpoint_path = '../../../../model/attention/zh_to_en/attention_ckpt'   # checkpoint保存路径。

    hidden_size = 1024  # LSTM的隐藏层规模。
    decoder_layers = 2  # 解码器中LSTM结构的层数。这个例子中编码器固定使用单层的双向LSTM。
    src_vocab_size = 4000  # 源语言词汇表大小。
    trg_vocab_size = 10000  # 目标语言词汇表大小。
    # --- 训练参数 ---
    batch_size = 100  # 训练数据batch的大小。
    num_epoch = 5  # 使用训练数据的轮数。
    keep_prob = 0.8  # 节点不被dropout的概率。
    max_grad_norm = 5  # 用于控制梯度膨胀的梯度大小上限。
    share_emb_and_softmax = True  # 在Softmax层和词向量层之间共享参数。

    max_len = 50  # 限定句子的最大单词数量。
    sos_id = 1  # 目标语言词汇表中<sos>的ID。


# --- 预测参数 ---
class para_predict2(object):
    checkpoint_path = '../../../../model/attention/zh_to_en/attention_ckpt-9000'

    hidden_size = 1024
    decoder_layers = 2
    src_vocab_size = 4000
    trg_vocab_size = 10000
    share_emb_and_softmax = True

    src_vocab = './data/zh.vocab'
    trg_vocab = './data/en.vocab'

    # 词汇表中<sos>和<eos>的id。在解码过程中需要用<sos>作为第一步的输入，并检查是否是<eos>，因此需要知道这两个符号的id
    sos_id = 1
    eos_id = 2

p_train2 = para_train2()
p_predict2 = para_predict2()