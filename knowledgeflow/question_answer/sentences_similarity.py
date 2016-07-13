#coding:utf8
'''
    Sentence Similarity Task
'''
import sys
import random
import logging
import numpy as np
import ConfigParser
from collections import OrderedDict

import tensorflow as tf
random.seed(2)
np.random.seed(9527)
sys.path.append('../..')
from utility import qa_utils
from utility.qa_utils import QaPairsTrain, QaPairsTest
from utility.utility import build_vocab, embedding_layer_weights, load_word2vec
from lcd import LCDBase, Conv1DLCD, Conv1DConcatLCD, margin_hinge, letor_binary_crossentropy
from apn import APNBase

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    stream=sys.stderr)

class SentenceSimilarityTask(object):
    def __init__(self, path):
        self.loadConfig(path)
        self.q_length = self.conf.getint('task', 'q_length')
        self.a_length = self.conf.getint('task', 'a_length')
        self.wdim = self.conf.getint('task', 'wdim')
        train = self.conf.get('task', 'train')
        print 'train path', train
        try:
            dev = self.conf.get('task', 'dev')
        except:
            dev = None
        test = self.conf.get('task', 'test')
        print 'test path', test
        try:
            self.predict = self.conf.get('task', 'predict')
        except:
            self.predict = None
        self.loadData(train, test=test, dev=dev)
        self.batch_size = self.conf.getint('train', 'batch_size')
        self.epochs = self.conf.getint('train', 'epochs')

    def loadConfig(self, path):
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(path)
        
    def loadData(self, train, test=None, dev=None):
        self.qapairs = OrderedDict()
        self.qapairs['train'] = QaPairsTrain(train)
        if test:
            self.qapairs['test'] = QaPairsTest(test)
        if dev:
            self.qapairs['dev'] = QaPairsTest(dev)
        self.qmax = max([qa.qmax for qa in self._availableParis()])
        self.amax = max([qa.amax for qa in self._availableParis()])
        print 'Q Length', self.qmax
        print 'A Length', self.amax
        self.data = []
        for name, pair in self.qapairs.items():
            self.data += pair.xq_data
            self.data += pair.xa_data
        self.reversed_vocab, self.vocabulary = build_vocab(self.data, start_with=['<PAD>'])
        map(lambda x: x.build(self.vocabulary, self.qmax, self.amax), self._availableParis())
        #self.qapairs['train'].shuffle()

    def remoteEmbedding(self):
        host = self.conf.get('embedding', 'host', 'szwg-rp-nlp349.szwg01.baidu.com')
        port = self.conf.getint('embedding', 'port')
        method = self.conf.get('embedding', 'method', 'word2vec')
        name = self.conf.get('embedding', 'name', 'en_google')
        import pymongo
        coll = pymongo.MongoClient(host=host,port=port)[method][name]
        word2vec = load_word2vec(coll, self.reversed_vocab)
        return embedding_layer_weights(self.reversed_vocab, word2vec, self.wdim)

    def equipModel(self):
        weights = self.remoteEmbedding()
        impl = APNBase(self.wdim)
        impl.readDefaultConfig(self.conf)
        impl.setEmbedding(weights[0])
        self.model = impl
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            tf.set_random_seed(1337)
            print self.sess.run(tf.random_uniform([1]))
            impl.build()
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(self.model.tensors['loss'])
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            self.sess.run(tf.initialize_all_variables())
            print self.sess.run(tf.random_uniform([1]))
    
    def train_step(self, xq_batch, xa_batch, y_batch):
        feed_dict = {
            self.model.tensors['q_input']: xq_batch,
            self.model.tensors['a_input']: xa_batch,
            self.model.tensors['label']: y_batch
        }
        _, step, loss, sparsity = self.sess.run(
            [self.train_op, self.global_step, self.model.tensors['loss']] + self.model.tensors['summary'], 
            feed_dict
        )
        if (step % 100 == 0):
            logging.info('LOSS %f @step %d sparsity %f' % (loss, step, sparsity))

    def test_step(self, xq_batch, xa_batch):
        feed_dict = {
            self.model.tensors['q_input']: xq_batch,
            self.model.tensors['a_input']: xa_batch,
        }
        predict = self.sess.run([self.model.tensors['similarity']], feed_dict)
        return predict

    def trainEpoch(self):
        self.qapairs['train'].partiteSamples()
        #self.qapairs['train'].shuffle()
        #self.qapairs['test'].shuffle()
        best_test_map = 0
        for _ in xrange(self.epochs):
            dev = self.qapairs.get('dev')
            test = self.qapairs.get('test')
            if dev:
                #MAP = dev.ndcg_score(self.model, k=20, batch_size=self.batch_size)
                MAP = dev.label_ranking_average_precision_score(lambda q,a: self.test_step(q,a), batch_size=self.batch_size)
                print('Dev MAP %f' % MAP)
            if test:
                #MAP = test.ndcg_score(self.model, k=20, batch_size=self.batch_size)
                MAP = test.label_ranking_average_precision_score(lambda q,a: self.test_step(q,a), batch_size=self.batch_size)
                print('Test MAP %f' % MAP)
                if MAP > best_test_map:
                    best_test_map = MAP
                    if self.predict: test.dumpResult(self.predict)
                    saver = tf.train.Saver(self.tensors['weights'], sharded=False)
            for xq, xa, y in self.qapairs['train'].pairwiseSampling(50):
                self.train_step(xq, xa, y)

        print('Best Test MAP %f' % best_test_map)

    def _availableParis(self):
        return self.qapairs.values()

def main():
    task = SentenceSimilarityTask('./configure/lcd_comments.conf')
    task.equipModel()
    task.trainEpoch()

if __name__ == '__main__':
    main()
