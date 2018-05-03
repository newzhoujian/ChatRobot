#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
from RNN import GRU
import numpy as np
import theano
from gensim.models.word2vec import Word2Vec
from PreProcess import WordVecs
from logistic_sgd import LogisticRegression
from Optimization import Adam
import theano.tensor as T
from SimAsImage import ConvSim
import sys
theano.config.floatX = 'float32'

max_turn = 10
def get_idx_from_sent_msg(sents, word_idx_map, max_l=50,mask = False):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    turns = []
    for sent in sents.split('_t_'):
        x = [0] * max_l
        x_mask = [0.] * max_l
        words = sent.split()
        length = len(words)
        for i, word in enumerate(words):
            if max_l - length + i < 0: continue
            if word in word_idx_map:
                x[max_l - length + i] = word_idx_map[word]
            x_mask[max_l - length + i] = 1
        if mask:
            x += x_mask
        turns.append(x)

    final = [0.] * (max_l * 2 * max_turn)
    for i in range(max_turn):
        if max_turn - i <= len(turns):
            for j in range(max_l * 2):
                final[i*(max_l*2) + j] = turns[-(max_turn-i)][j]

    return final

def get_idx_from_sent(sent, word_idx_map, max_l=50,mask=False):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = [0] * max_l
    x_mask = [0.] * max_l
    words = sent.split()
    length = len(words)
    for i, word in enumerate(words):
        if max_l - length + i < 0: continue
        if word in word_idx_map:
            x[max_l - length + i] = word_idx_map[word]
        x_mask[max_l - length + i] = 1
    if mask:
        x += x_mask
    return x

def _dropout_from_layer(rng, layer, p):
    """
    p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def predict(datasets, U,
            n_epochs=5, batch_size=20, max_l=100, hidden_size=200, word_embedding_size=200,
            session_hidden_size=50, session_input_size =50, model_name='SMN_last.bin'):
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    hiddensize = hidden_size
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    lsize, rsize = max_l, max_l

    sessionmask = T.matrix()
    lx = []
    lxmask = []
    for i in range(max_turn):
        lx.append(T.matrix())
        lxmask.append(T.matrix())

    index = T.lscalar()
    rx = T.matrix('rx')
    rxmask = T.matrix()
    y = T.ivector('y')
    Words = theano.shared(value=U, name="Words")
    llayer0_input = []
    for i in range(max_turn):
        llayer0_input.append(Words[T.cast(lx[i].flatten(), dtype="int32")]
                             .reshape((lx[i].shape[0], lx[i].shape[1], Words.shape[1])))

    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],rx.shape[1],Words.shape[1]))


    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]

    train_set_lx = []
    train_set_lx_mask = []
    q_embedding = []
    offset = 2 * lsize
    for i in range(max_turn):
        train_set_lx.append(theano.shared(np.asarray(train_set[:, offset*i:offset*i+lsize]
                                                     , dtype=theano.config.floatX), borrow=True))
        train_set_lx_mask.append(theano.shared(np.asarray(train_set[:, offset*i+lsize:offset*i+2*lsize]
                                                          , dtype=theano.config.floatX), borrow=True))
    train_set_rx = theano.shared(np.asarray(train_set[:, offset*max_turn:offset*max_turn+lsize]
                                            , dtype=theano.config.floatX), borrow=True)
    train_set_rx_mask = theano.shared(np.asarray(train_set[:, offset*max_turn+lsize:offset*max_turn+2*lsize]
                                                 , dtype=theano.config.floatX), borrow=True)
    train_set_session_mask = theano.shared(np.asarray(train_set[:, -max_turn-1:-1]
                                                      , dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(np.asarray(train_set[:, -1], dtype="int32"), borrow=True)

    val_set_lx = []
    val_set_lx_mask = []
    for i in range(max_turn):
        val_set_lx.append(theano.shared(np.asarray(dev_set[:, offset*i:offset*i + lsize]
                                                   , dtype=theano.config.floatX), borrow=True))
        val_set_lx_mask.append(theano.shared(np.asarray(dev_set[:, offset*i+lsize:offset*i+2*lsize]
                                                        , dtype=theano.config.floatX), borrow=True))
    val_set_rx = theano.shared(np.asarray(dev_set[:, offset*max_turn:offset*max_turn + lsize],
                                          dtype=theano.config.floatX), borrow=True)
    val_set_rx_mask = theano.shared(np.asarray(dev_set[:, offset*max_turn+lsize:offset*max_turn+2*lsize],
                                               dtype=theano.config.floatX), borrow=True)
    val_set_session_mask = theano.shared(np.asarray(dev_set[:, -max_turn-1:-1]
                                                    , dtype=theano.config.floatX), borrow=True)
    val_set_y =theano.shared(np.asarray(dev_set[:, -1], dtype="int32"), borrow=True)

    dic = {}
    for i in range(max_turn):
        dic[lx[i]] = train_set_lx[i][index*batch_size:(index+1)*batch_size]
        dic[lxmask[i]] = train_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    dic[rx] = train_set_rx[index*batch_size:(index+1)*batch_size]
    dic[sessionmask] = train_set_session_mask[index*batch_size:(index+1)*batch_size]
    dic[rxmask] = train_set_rx_mask[index*batch_size:(index+1)*batch_size]
    dic[y] = train_set_y[index*batch_size:(index+1)*batch_size]

    val_dic = {}
    for i in range(max_turn):
        val_dic[lx[i]] = val_set_lx[i][index*batch_size:(index+1)*batch_size]
        val_dic[lxmask[i]] = val_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    val_dic[rx] = val_set_rx[index*batch_size:(index+1)*batch_size]
    val_dic[sessionmask] = val_set_session_mask[index*batch_size:(index+1)*batch_size]
    val_dic[rxmask] = val_set_rx_mask[index*batch_size:(index+1)*batch_size]
    val_dic[y] = val_set_y[index*batch_size:(index+1)*batch_size]

    sentence2vec = GRU(n_in=word_embedding_size, n_hidden=hiddensize, n_out=hiddensize)

    for i in range(max_turn):
        q_embedding.append(sentence2vec(llayer0_input[i], lxmask[i], True))
    r_embedding = sentence2vec(rlayer0_input, rxmask, True)

    pooling_layer = ConvSim(rng, max_l, session_input_size, hidden_size=hiddensize)

    poolingoutput = []

    for i in range(max_turn):
        poolingoutput.append(pooling_layer(llayer0_input[i], rlayer0_input,
                                           q_embedding[i], r_embedding))


    session2vec = GRU(n_in=session_input_size, n_hidden=session_hidden_size, n_out=session_hidden_size)

    res = session2vec(T.stack(poolingoutput, 1), sessionmask)
    classifier = LogisticRegression(res, session_hidden_size, 2, rng)

    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    opt = Adam()
    params = classifier.params
    params += sentence2vec.params
    params += session2vec.params
    params += pooling_layer.params
    params += [Words]

    load_params(params, model_name)

    predict = classifier.predict_prob

    val_model = theano.function([index], [y, predict, cost, error], givens=val_dic
                                , on_unused_input='ignore')
    f = open('result.txt', 'w')
    loss = 0.
    for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
        a, b, c, d = val_model(minibatch_index)
        print c
        loss += c
        for i in range(batch_size):
            f.write(str(b[i][1]))
            f.write('\t')
            f.write(str(a[i]))
            f.write('\n')
    print loss/(datasets[1].shape[0]/batch_size)


def load_params(params,filename):
    f = open(filename)
    num_params = cPickle.load(f)
    for p,w in zip(params,num_params):
        p.set_value(w.astype('float32'),borrow=True)
    print "load successfully"

def train(datasets, U,
          n_epochs=5, batch_size=20, max_l=50, hidden_size=200, word_embedding_size=200,
          session_hidden_size=50, session_input_size=50, model_name='SMN_last.bin'):
    # 设置hiddensize，lsize，rsize，初始化，定义符号化编程的步骤
    hiddensize = hidden_size
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    lsize, rsize = max_l,max_l
    sessionmask = T.matrix()
    lx = []
    lxmask = []
    for i in range(max_turn):
        lx.append(T.matrix())
        lxmask.append(T.matrix())

    index = T.lscalar()
    rx = T.matrix('rx')
    rxmask = T.matrix()
    y = T.ivector('y')
    Words = theano.shared(value=U, name="Words")

    """
    第一层GRU的输入，llayer0_input和rlayer0_input
    llayer0_input是多轮对话构成的词级的向量，list类型，len=10，每一条是(200,50,200)的矩阵；
    rlayer0_input是答句构成的词级的向量，(200,50,200)的矩阵
    """
    llayer0_input = []
    for i in range(max_turn):
        llayer0_input.append(Words[T.cast(lx[i].flatten(),dtype="int32")]
                             .reshape((lx[i].shape[0], lx[i].shape[1], Words.shape[1])))

    rlayer0_input = Words[T.cast(rx.flatten(), dtype="int32")].reshape((rx.shape[0], rx.shape[1], Words.shape[1]))

    # 从datasets中获取训练集，验证集和测试集
    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]

    train_set_lx = []
    train_set_lx_mask = []
    q_embedding = []
    offset = 2 * lsize

    """
    划分训练集的lx和rx，即train_set_lx，train_set_rx
    划分lxmask，rxmask，sessionmask和y
    """
    for i in range(max_turn):
        print(offset*i, offset*i + lsize)
        train_set_lx.append(theano.shared(np.asarray(train_set[:, offset*i:offset*i + lsize]
                                                     , dtype=theano.config.floatX), borrow=True))
        train_set_lx_mask.append(theano.shared(np.asarray(train_set[:, offset*i+lsize:offset*i+2*lsize]
                                                          , dtype=theano.config.floatX), borrow=True))

    train_set_rx = theano.shared(np.asarray(train_set[:, offset*max_turn:offset*max_turn + lsize]
                                            , dtype=theano.config.floatX), borrow=True)
    train_set_rx_mask = theano.shared(np.asarray(train_set[:, offset*max_turn+lsize:offset*max_turn+2*lsize]
                                                 , dtype=theano.config.floatX), borrow=True)
    train_set_session_mask = theano.shared(np.asarray(train_set[:, -max_turn-1:-1]
                                                      , dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(np.asarray(train_set[:, -1], dtype="int32"), borrow=True)

    val_set_lx = []
    val_set_lx_mask = []
    """
    验证集同训练集
    """
    for i in range(max_turn):
        val_set_lx.append(theano.shared(np.asarray(dev_set[:, offset*i:offset*i + lsize]
                                                   , dtype=theano.config.floatX), borrow=True))
        val_set_lx_mask.append(theano.shared(np.asarray(dev_set[:, offset*i+lsize:offset*i+2*lsize]
                                                        , dtype=theano.config.floatX), borrow=True))

    val_set_rx = theano.shared(np.asarray(dev_set[:, offset*max_turn:offset*max_turn+lsize],
                                          dtype=theano.config.floatX), borrow=True)
    val_set_rx_mask = theano.shared(np.asarray(dev_set[:, offset*max_turn+lsize:offset*max_turn+2*lsize],
                                               dtype=theano.config.floatX), borrow=True)
    val_set_session_mask = theano.shared(np.asarray(dev_set[:, -max_turn-1:-1]
                                                    , dtype=theano.config.floatX), borrow=True)
    val_set_y =theano.shared(np.asarray(dev_set[:, -1], dtype="int32"), borrow=True)

    """
    构造训练集字典输入
    """
    dic = {}
    for i in range(max_turn):
        dic[lx[i]] = train_set_lx[i][index*batch_size:(index+1)*batch_size]
        dic[lxmask[i]] = train_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    dic[rx] = train_set_rx[index*batch_size:(index+1)*batch_size]
    dic[sessionmask] = train_set_session_mask[index*batch_size:(index+1)*batch_size]
    dic[rxmask] = train_set_rx_mask[index*batch_size:(index+1)*batch_size]
    dic[y] = train_set_y[index*batch_size:(index+1)*batch_size]

    """
    构造验证集字典输入
    """
    val_dic = {}
    for i in range(max_turn):
        val_dic[lx[i]] = val_set_lx[i][index*batch_size:(index+1)*batch_size]
        val_dic[lxmask[i]] = val_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    val_dic[rx] = val_set_rx[index*batch_size:(index+1)*batch_size]
    val_dic[sessionmask] = val_set_session_mask[index*batch_size:(index+1)*batch_size]
    val_dic[rxmask] = val_set_rx_mask[index*batch_size:(index+1)*batch_size]
    val_dic[y] = val_set_y[index*batch_size:(index+1)*batch_size]

    """
    第一次GRU，q_embedding和r_embedding是输出的结果
    q_embedding是多轮对话构成的句子级的向量，list类型，len=10，每一条是(200,50,200)的矩阵；
    rlayer0_input是答句构成的句子级的向量，(200,50,200)的矩阵
    """
    sentence2vec = GRU(n_in=word_embedding_size, n_hidden=hiddensize, n_out=hiddensize)

    for i in range(max_turn):
        q_embedding.append(sentence2vec(llayer0_input[i], lxmask[i], True))
    r_embedding = sentence2vec(rlayer0_input, rxmask, True)

    pooling_layer = ConvSim(rng, max_l, session_input_size, hidden_size=hiddensize)

    """
    卷积，输入是llayer0_input和rlayer0_input，q_embedding和r_embedding
    过程是llayer0_input中的每一条数据和rlayer0_input做卷积
    q_embedding中的每一条数据和r_embedding做卷积，合并结果
    输出是poolingoutput，list类型，len=10，每一条是(200,50)的矩阵；
    """
    poolingoutput = []

    for i in range(max_turn):
        poolingoutput.append(pooling_layer(llayer0_input[i], rlayer0_input,
                                           q_embedding[i], r_embedding))

    con_test = theano.function([index], pooling_layer(llayer0_input[0], rlayer0_input,
                                           q_embedding[0], r_embedding), givens=dic, on_unused_input='ignore')
    print con_test(0).shape
    sys.exit()

    """
    第二次GRU，输入是poolingoutput
    输出是res，(200,50)的矩阵，用一个50维的向量表示一个session
    """
    session2vec = GRU(n_in=session_input_size, n_hidden=session_hidden_size, n_out=session_hidden_size)

    res = session2vec(T.stack(poolingoutput, 1), sessionmask)

    """
    逻辑回归分类：输入是res，标签是y
    损失是negative_log_likelihood
    """
    classifier = LogisticRegression(res, session_hidden_size, 2, rng)

    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)

    opt = Adam()
    params = classifier.params
    params += sentence2vec.params
    params += session2vec.params
    params += pooling_layer.params
    params += [Words]

    grad_updates = opt.Adam(cost=cost, params=params, lr=0.001)

    # 训练模型的定义
    train_model = theano.function([index], cost, updates=grad_updates, givens=dic, on_unused_input='ignore')

    # 验证模型的定义
    val_model = theano.function([index], [cost, error], givens=val_dic, on_unused_input='ignore')
    best_dev = 1.

    n_train_batches = datasets[0].shape[0]/batch_size

    """
    开始灌数据输入，n_epochs是训练次数
    """
    for i in xrange(n_epochs):
        cost = 0
        total = 0.
        # 训练的代码
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_cost = train_model(minibatch_index)
            print "minibatch_index"
            print minibatch_index
            total = total + 1
            cost = cost + batch_cost
            if total % 2 == 0:
                print total, cost/total
        cost = cost / n_train_batches
        print "echo %d loss %f" % (i, cost)

        # 验证的代码
        cost = 0
        errors = 0
        j = 0
        for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
            tcost, terr = val_model(minibatch_index)
            cost += tcost
            errors += terr
            j = j+1
        cost = cost / j
        errors = errors / j
        if cost < best_dev:
            best_dev = cost
            save_params(params,model_name)
        print "echo %d dev_loss %f" % (i, cost)
        print "echo %d dev_accuracy %f" % (i, 1 - errors)

def save_params(params,filename):
    num_params = [p.get_value() for p in params]
    f = open(filename, 'wb')
    cPickle.dump(num_params, f)

def get_session_mask(sents):
    session_mask = [0.] * max_turn
    turns = []
    for sent in sents.split('_t_'):
        words = sent.split()
        if len(words) > 0:
            turns.append(len(words))

    for i in range(max_turn):
        if max_turn - i <= len(turns):
            session_mask[-(max_turn-i)] = 1.
    return session_mask

def make_data(revs, word_idx_map, max_l=50, filter_h=3, val_test_splits=[2,3],validation_num = 200):
    """
    Transforms sentences into a 2-d matrix.
    """
    # validation_num为验证集的大小
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent_msg(rev["m"], word_idx_map, max_l, True)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, True)
        sent += get_session_mask(rev["m"])
        sent.append(float(rev["y"]))
        if len(val) >= validation_num:
            train.append(sent)
        else:
            val.append(sent)

    train = np.array(train, dtype="int")
    val = np.array(val, dtype="int")
    test = np.array(test, dtype="int")
    print 'trainning data', len(train), 'val data', len(val)

    return [train, val, test]



if __name__=="__main__":
    train_flag = True  # 控制当前任务是训练还是测试，train_flag = True的时候做训练
    max_word_per_utterence = 50  # 一句话中最大的单词数不超过50个
    dataset = r"../processeddata/ubuntu_data_temp.mul.train"  # 训练集的数据库

    # 获取dataset中的数据
    x = cPickle.load(open(dataset, "rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]

    if not train_flag:
        x = cPickle.load(open(r"ubuntu_data_temp.mul.test", "rb"))
        revs, wordvecs2, max_l2 = x[0], x[1], x[2]

    # 划分了训练集、验证集、测试集
    datasets = make_data(revs, wordvecs.word_idx_map, max_l=max_word_per_utterence)

    if train_flag:  # batchsize设置为200，hidden_size和word_embedding_size设置为200
        train(datasets, wordvecs.W, batch_size=200, max_l=max_word_per_utterence
              , hidden_size=200, word_embedding_size=200)
    else:
        predict(datasets,wordvecs.W, batch_size=200, max_l=max_word_per_utterence
                , hidden_size=200, word_embedding_size=200)

