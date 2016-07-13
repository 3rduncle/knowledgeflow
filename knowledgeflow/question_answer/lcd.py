#coding:utf8
'''
    Sentence Similarity Task
'''
import ConfigParser
import numpy as np
from collections import OrderedDict

import theano
import keras
from theano import tensor as T
from keras import backend as K
from theano.compile.nanguardmode import NanGuardMode
from keras.preprocessing import sequence
from keras.models import Graph, Model
from keras.constraints import maxnorm
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam,SGD
import keras.initializations as initializations
import keras.regularizers as regularizers
from keras.layers.advanced_activations import ELU, PReLU, SReLU

SAFE_EPSILON = 1e-20

def margin_hinge(y_true, y_pred, margin=0.5):
    # y_pred are the dot product similarities, in interleaved form (positive example, negative example, ...)
    # y_true is simply 1, 0, 1, 0
    signed = 2 * y_pred * (y_true - 0.5) # we do this, just so that y_true is part of the computational graph
    pos = signed[0::2]
    neg = signed[1::2]
    # negative samples are multiplied by -1, so that the sign in the rankSVM objective is flipped below
    rank_hinge_loss = K.mean(K.relu(margin - pos - neg))
    return rank_hinge_loss

def letor_binary_crossentropy(y_true, y_pred):
    signed = 2 * y_pred * (y_true - 0.5)
    pos = signed[0::2]
    neg = signed[1::2]
    s = pos - neg
    es = K.exp(p)
    p = es / (1 + es)
    return K.mean(K.binary_crossentropy(p, y_true), axis=-1)

# 注意自定义的merge_mode接收到的tensor带有额外的batch_size，
# 即这一层接收到的tensor的ndim=(batch_size, row, col)
# 至于3d矩阵如何按照样本分别求内积就说来话长了。。看下面的用法
# http://deeplearning.net/software/theano/library/tensor/basic.html
def semantic_matrix(argv):
    assert len(argv) == 2
    q = argv[0]
    a = argv[1]
    q_sqrt = K.sqrt((q ** 2).sum(axis=2, keepdims=True))
    a_sqrt = K.sqrt((a ** 2).sum(axis=2, keepdims=True))
    denominator = K.batch_dot(q_sqrt, K.permute_dimensions(a_sqrt, [0,2,1]))
    return K.batch_dot(q, K.permute_dimensions(a, [0,2,1])) / (denominator + SAFE_EPSILON)

# 注意idx是二维的矩阵
# 如何执行类似batch index的效果折腾了半天
# 参考https://groups.google.com/forum/#!topic/theano-users/7gUdN6E00Dc
# 注意argmax里面是2 - axis
# 注意theano里面a > 0返回的是整数类似[1,1,0]的矩阵，里面的值是整数而不
# 是bool值，不能直接作为索引值
# 因此需要做这样的操作T.set_subtensor(ib[(ib < 0).nonzero()], 0)
def match_matrix(vectors, match, axis=0, w=3):
    # if axis = 0 
    # source_length = amax
    # target_length = qmax 
    # results shape=(batch_size, qmax, wdim)
    # vectors shape=(batch_size, amax, wdim)
    # match   shape=(batch_size, qmax, amax)
    batch_size, qmax, amax = match.shape
    _, _, wdim = vectors.shape
    if axis == 0:
        source_length = amax
        target_length = qmax
        dims = [0,1,2]
    elif axis == 1:
        source_length = qmax
        target_length = amax
        dims = [0,2,1]
    match = K.permute_dimensions(match, dims)
    source_length = (qmax, amax)[1 - axis]
    target_length = (qmax, amax)[axis]
    m = source_length - 1
    batched_length = batch_size * target_length
    # reshaped match shape=(batch_size * qmax, amax)
    batched_match = match.reshape((batched_length, source_length))
    # shape=(batch_size * qmax,), range in [0,1]
    value = batched_match.max(axis=1)
    # shape=(batch_size * qmax, ), range in [0, amax) 
    index = batched_match.argmax(axis=1)
    params = []
    params.append((value, index))
    for j in range(1, w + 1):
        ib = index - j
        ibs = T.set_subtensor(ib[(ib < 0).nonzero()], 0)
        iu = index + j
        ius = T.set_subtensor(iu[(iu > m).nonzero()], m)
        params.append((batched_match[T.arange(batched_length), ibs], ibs))
        params.append((batched_match[T.arange(batched_length), ius], ius))
    i0 = T.repeat(T.arange(batch_size), target_length).flatten()
    indexed = 0
    weights = 0
    for value, index in params:
        # shape=(batch_size * qmax,) => shape=(batch_size * qmax, 1) 
        value = K.expand_dims(value, 1)
        # shape=(batch_size * qmax, wdim)
        indexed += vectors[i0, index, :] * value
        weights += value
    results = (indexed / weights).reshape((batch_size, target_length, wdim))
    return results

def parallel(source, target):
    einner_product = (source * target).sum(axis=2).reshape((source.shape[0],source.shape[1], 1))
    enorm = (target ** 2).sum(axis=2).reshape((source.shape[0],source.shape[1],1))
    response = target * einner_product / (enorm + SAFE_EPSILON)
    return response
# merge q_lembeding, q_match -> two channel matrix
def decomposite(source, target):
    q_pos = parallel(source, target)
    q_neg = source - q_pos
    # shape=(2, batch_size, q_length, wdim)
    channels = T.stacklists([q_pos, q_neg])
    return K.permute_dimensions(channels, [1,0,2,3])

def compute_similar(q, a):
    q_sqrt = K.sqrt((q ** 2).sum(axis=1))
    a_sqrt = K.sqrt((a ** 2).sum(axis=1))
    denominator = q_sqrt * a_sqrt
    output = (q * a).sum(axis=1) / (denominator + SAFE_EPSILON)
    return K.expand_dims(output, 1)

class LCDBase(object):
    '''
    Reference: Sentence Similarity Learning by Lexical Decomposition and Composition[http://arxiv.org/pdf/1602.07019v1.pdf]
    '''
    def __init__(self, q_length, a_length, wdim=300):
        self.q_length = q_length
        self.a_length = a_length
        self.wdim = wdim
        self.params = {}
        self.layers = {}
        self.tensors = {}
        self.embedding_params = {}
        self.built = False

    def readDefaultConfig(self, conf):
        # model params
        self.params['window'] = conf.getint('model', 'window')
        self.params['filters'] = map(int, conf.get('model', 'filters').split(','))
        self.params['nb_filter'] = conf.getint('model', 'nb_filter')
        self.params['w_maxnorm'] = conf.getint('model', 'w_maxnorm')
        self.params['b_maxnorm'] = conf.getint('model', 'b_maxnorm')
        self.params['dropout'] = conf.getfloat('model', 'dropout')
        # train parmas
        self.params['batch_size'] = conf.getint('train', 'batch_size')
        self.params['epochs'] = conf.getint('train', 'epochs')

    def setConfig(self, key, val):
        self.params[key] = val

    def buildInput(self):
        # Question Network Input
        q_input = Input(name='q_input', shape=(self.q_length,), dtype='int32')
        # Answer Network Input
        a_input = Input(name='a_input', shape=(self.a_length,), dtype='int32')
        self.tensors['q_input'] = q_input
        self.tensors['a_input'] = a_input
        
    def setEmbedding(self, weights):
        self.embedding_params['weights'] = weights
 
    def buildEmbedding(self, name):
        weights = self.embedding_params.get('weights')
        assert weights
        self.layers[name] = Embedding(
            weights[0].shape[0],
            weights[0].shape[1],
            weights = weights,
            trainable = self.params.get('embedding_trainable', False),
            name=name
        )
    def sharedEmbedding(self):
        self.buildEmbedding('shared-embedding')

    def sepreatedEmbedding(self):
        self.buildEmbedding('q-embedding')
        self.buildEmbedding('a-embedding')
        
    def buildComposition(self, shared=True):
        q_input = self.tensors['q_input']
        a_input = self.tensors['a_input']
        if shared:
            q_embedding = self.layers['shared-embedding'](q_input)
            a_embedding = self.layers['shared-embedding'](a_input)
        else:
            q_embedding = self.layers['q-embedding'](q_input)
            a_embedding = self.layers['a-embedding'](a_input)

        print('Embedding ndim q %d a %d' % (K.ndim(q_embedding), K.ndim(a_embedding)))
        print('Embedding shape ', q_embedding._keras_shape, a_embedding._keras_shape)

        # compute Semantic Matching
        cross = Merge(
        #   [q_embedding, a_embedding],
            mode=semantic_matrix,
            output_shape=(self.q_length, self.a_length),
            name='semantic'
        )
        semantic = cross([q_embedding, a_embedding])
        print('Semantic ndim %d' % K.ndim(semantic))
        print('Semantic shape ', semantic._keras_shape)
        print('Semantic shape ', cross.get_output_shape_at(0))

        # compute cross 
        q_match = merge(
            [a_embedding, semantic],
            mode=lambda x: match_matrix(*x,axis=0, w=self.params['window']),
            output_shape=(self.q_length, self.wdim),
            name='q_match'
        )
        print('q_match ', q_match._keras_shape, K.ndim(q_match))

        a_match = merge(
            [q_embedding, semantic],
            mode=lambda x: match_matrix(*x,axis=1, w=self.params['window']),
            output_shape=(self.a_length, self.wdim),
            name='a_match'
        )
        print('Match ndim q %d a %d' % (K.ndim(q_match), K.ndim(a_match)))
        print('Match shape ', q_match._keras_shape, a_match._keras_shape)
        self.tensors['q-embedding'] = q_embedding
        self.tensors['a-embedding'] = a_embedding
        self.tensors['q-match'] = q_match
        self.tensors['a-match'] = a_match

    def buildDecomposition(self):
        q_embedding = self.tensors['q-embedding']
        a_embedding = self.tensors['a-embedding']
        q_match = self.tensors['q-match']
        a_match = self.tensors['a-match']
        # compute q+, q-, a+, a-
        # 注意为什么其他的层不需要加BATCH_SIZE，而这里却突然需要了呢？
        # 原因Lambda的坑，Lambda的ouput_shape不需要指定BATCH_SIZE，会
        # 自行推导：当Lambda的上层输出中含有BATCH_SIZE时，使用改值作
        # 为本层的BATCH_SIZE，如果没有时我就呵呵了，不知道是怎么推的。
        # 因此这层Merge给定BATCH_SIZE是填下层Lambda的坑
        q_channels = Merge(
            mode=lambda x: decomposite(*x),
            output_shape=(self.params['batch_size'], 2, self.q_length, self.wdim),
            name='q-channels'
        )([q_embedding, q_match])

        a_channels = Merge(
            mode=lambda x: decomposite(*x),
            output_shape=(self.params['batch_size'], 2, self.a_length, self.wdim),
            name='a-channels',
        )([a_embedding, a_match])
        print('q_channels', q_channels._keras_shape, K.ndim(q_channels))
        print('a_channels', a_channels._keras_shape, K.ndim(a_channels))
        self.tensors['q-channels'] = q_channels
        self.tensors['a-channels'] = a_channels

    def buildConvolution(self, name):
        filters = self.params.get('filters')
        nb_filter = self.params.get('nb_filter')
        assert filters
        assert nb_filter
        convs = []
        for fsz in filters:
            layer_name = '%s-conv-%d' % (name, fsz)
            conv = Convolution2D(
                nb_filter=nb_filter,
                nb_row=fsz,
                nb_col=self.wdim,
                border_mode='valid',
                init='glorot_uniform',
                W_constraint=maxnorm(self.params.get('w_maxnorm')),
                b_constraint=maxnorm(self.params.get('b_maxnorm')),
                name=layer_name
            )
            convs.append(conv)
        self.layers['%s-convolution' % name] = convs

    def sharedConvolution(self):
        self.buildConvolution('shared')

    def doubleConvolution(self):
        self.buildConvolution('q')
        self.buildConvolution('a')

    def quarterConvolution(self):
        self.buildConvolution('q-')
        self.buildConvolution('q+')
        self.buildConvolution('a+')
        self.buildConvolution('a-')

    def linkFeature(self, input_name, conv_name, activation='tanh'):
        print('Am I called')
        filters = self.params.get('filters')
        nb_filter = self.params.get('nb_filter')
        convs = self.layers.get(conv_name)
        assert filters
        assert convs
        features = []
        for fsz, conv in zip(filters, convs):
            conv_output = conv(self.tensors[input_name])
            if type(activation) == type(''):
                act = Activation(
                    activation, name='%s-act-%d' % (input_name, fsz)
                )(conv_output)
            else:
                act = activation(
                    name='%s-advanced-act-%d' % (input_name, fsz)
                )(conv_output)
            maxpool = Lambda(
                lambda x: K.max(x[:,:,:,0], axis=2),
                output_shape=(nb_filter,),
                name='%s-maxpool-%d' % (input_name, fsz)
            )(act)
            features.append(maxpool)
        if len(features) > 1:
            return Merge(mode='concat', name='%s-feature' % input_name)(features)
        else:
            return features[0]

    def buildFeatures(self, type='shared'):
        assert self.checkTensor('q-channels')
        assert self.checkTensor('a-channels')
        srelu = lambda name: SReLU(name=name)
        features = []
        if type == 'shared':
            q_features = self.linkFeature('q-channels', 'shared-convolution', activation='tanh')
            a_features = self.linkFeature('a-channels', 'shared-convolution', activation='tanh')
        else:
            raise Error('Not Supported')
        print('q-features', q_features._keras_shape, K.ndim(q_features))
        print('a-features', a_features._keras_shape, K.ndim(a_features))
        self.tensors['q-features'] = q_features
        self.tensors['a-features'] = a_features

    def buildSimilarity(self):
        q_features = self.tensors['q-features']
        a_features = self.tensors['a-features']
        self.tensors['similarity'] = merge(
            [q_features, a_features],
            mode=lambda x: compute_similar(*x),
            output_shape=(self.params['batch_size'],1),
            name='similarity'
        )

    def buildSigmoid(self):
        q_features = self.tensors['q-features']
        a_features = self.tensors['a-features']
        concat = merge(
            [q_features, a_features],
            mode='concat',
            name='concat'
        )
        dropout = self.params.get('dropout',0.0)
        fc = Dense(1, name='fc')(Dropout(dropout)(concat))
        sigmoid = Activation('sigmoid', name='sigmoid')(Dropout(0.0)(fc))
        self.tensors['sigmoid'] = sigmoid

    def build(self):
        self.buildInput()
        self.sharedEmbedding()
        self.buildComposition(shared=True)
        self.sharedConvolution()
        self.buildDecomposition()
        self.buildFeatures(type='shared')
        self.buildSimilarity()
        self.buildSigmoid()
        self.built = True

    def buildModel(self, input, output, loss='binary_crossentropy'):
        assert self.built
        inputs = [self.tensors[i] for i in input]
        outputs = [self.tensors[i] for i in output]
        model = Model(input=inputs, output=outputs)
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        return model

    def checkTensor(self, name):
        return name in self.tensors

    def checkLayer(self, name):
        return name in self.layers

class Conv1DLCD(LCDBase):
    def buildDecomposition(self):
        q_embedding = self.tensors['q-embedding']
        a_embedding = self.tensors['a-embedding']
        q_match = self.tensors['q-match']
        a_match = self.tensors['a-match']
        # compute q+, q-, a+, a-
        # 注意为什么其他的层不需要加BATCH_SIZE，而这里却突然需要了呢？
        # 原因Lambda的坑，Lambda的ouput_shape不需要指定BATCH_SIZE，会
        # 自行推导：当Lambda的上层输出中含有BATCH_SIZE时，使用改值作
        # 为本层的BATCH_SIZE，如果没有时我就呵呵了，不知道是怎么推的。
        # 因此这层Merge给定BATCH_SIZE是填下层Lambda的坑
        q_pos = Merge(
            mode=lambda x: parallel(*x),
            output_shape=(self.params['batch_size'], self.q_length, self.wdim),
            name='q+'
        )([q_embedding, q_match])

        # 注意这里不能直接用1 - q_pos获取，否则会丢掉_keras_shape属性
        # 注意这里的output_shape是不需要给batch_size的和Merge不同
        q_neg = Merge(
            mode=lambda x: x[0] - x[1],
            output_shape=(self.params['batch_size'], self.q_length, self.wdim),
            name='q-'
        )([q_embedding, q_pos])
        print('q_pos', q_pos._keras_shape, K.ndim(q_pos))
        print('q_neg', q_neg._keras_shape, K.ndim(q_neg))

        a_pos = Merge(
            mode=lambda x: parallel(*x),
            output_shape=(self.params['batch_size'], self.a_length, self.wdim),
            name='a+',
        )([a_embedding, a_match])
        a_neg = Merge(
            mode=lambda x: x[0] - x[1],
            output_shape=(self.params['batch_size'], self.a_length, self.wdim),
            name='a-'
        )([a_embedding, a_pos])
        print('a_pos', a_pos._keras_shape, K.ndim(a_pos))
        print('a_neg', a_neg._keras_shape, K.ndim(a_neg))
        self.tensors['q+'] = q_pos
        self.tensors['q-'] = q_neg
        self.tensors['a+'] = a_pos
        self.tensors['a-'] = a_neg

    def buildConvolution(self, name):
        filters = self.params.get('filters')
        nb_filter = self.params.get('nb_filter')
        assert filters
        assert nb_filter
        convs = []
        for fsz in filters:
            layer_name = '%s-conv-%d' % (name, fsz)
            conv = Convolution1D(
                nb_filter=nb_filter,
                filter_length=fsz,
                border_mode='valid',
                #activation='relu',
                subsample_length=1,
                init='glorot_uniform',
                #init=init,
                #init=lambda shape, name: initializations.uniform(shape, scale=0.01, name=name),
                W_constraint=maxnorm(self.params.get('w_maxnorm')),
                b_constraint=maxnorm(self.params.get('b_maxnorm')),
                #W_regularizer=regularizers.l2(self.params.get('w_l2')),
                #b_regularizer=regularizers.l2(self.params.get('b_l2')),
                #input_shape=(self.q_length, self.wdim),
                name=layer_name
            )
            convs.append(conv)
        self.layers['%s-convolution' % name] = convs

    def doubleFeature(self, pos, neg, conv_name, activation='tanh'):
        name = '%s+%s' % (pos, neg)
        filters = self.params['filters']
        nb_filter = self.params['nb_filter']
        convs = self.layers[conv_name]
        features = []
        pos = self.tensors[pos]
        neg = self.tensors[neg]
        for fsz, conv in zip(filters, convs):
            sum = Merge(
                mode='sum',
            )([conv(pos), conv(neg)])
            if type(activation) == type(''):
                act = Activation(
                    activation, name='%s-act-%d' % ('+'.join(input_names), fsz)
                )(sum)
            else:
                act = activation(
                    name='%s-advanced-act-%d' % (name, fsz)
                )(sum)
            maxpool = Lambda(
                lambda x: K.max(x, axis=1),
                output_shape=(nb_filter,),
                name='%s-maxpool-%d' % (name, fsz)
            )(act)
            print('maxpool', maxpool._keras_shape)
            features.append(maxpool)
        if len(features) > 1:
            return Merge(
                mode='concat', 
                name='%s-feature' % name,
            )(features)
        else:
            return features[0]

    def buildFeatures(self, type='shared'):
        assert self.checkTensor('q+')
        assert self.checkTensor('q-')
        assert self.checkTensor('a+')
        assert self.checkTensor('a-')
        srelu = lambda name: SReLU(name=name)
        if type == 'shared':
            q_features = self.doubleFeature('q+', 'q-', 'shared-convolution', activation=srelu)
            a_features = self.doubleFeature('a+', 'a-', 'shared-convolution', activation=srelu)
        else:
            raise Error('Not Supported')
        print('q-features', q_features._keras_shape)
        print('a-features', a_features._keras_shape)
        self.tensors['q-features'] = q_features
        self.tensors['a-features'] = a_features

class Conv1DConcatLCD(LCDBase):

    def buildDecomposition(self):
        q_embedding = self.tensors['q-embedding']
        a_embedding = self.tensors['a-embedding']
        q_match = self.tensors['q-match']
        a_match = self.tensors['a-match']
        # compute q+, q-, a+, a-
        # 注意为什么其他的层不需要加BATCH_SIZE，而这里却突然需要了呢？
        # 原因Lambda的坑，Lambda的ouput_shape不需要指定BATCH_SIZE，会
        # 自行推导：当Lambda的上层输出中含有BATCH_SIZE时，使用改值作
        # 为本层的BATCH_SIZE，如果没有时我就呵呵了，不知道是怎么推的。
        # 因此这层Merge给定BATCH_SIZE是填下层Lambda的坑
        q_pos = Merge(
            mode=lambda x: parallel(*x),
            output_shape=(self.params['batch_size'], self.q_length, self.wdim),
            name='q+'
        )([q_embedding, q_match])

        # 注意这里不能直接用1 - q_pos获取，否则会丢掉_keras_shape属性
        # 注意这里的output_shape是不需要给batch_size的和Merge不同
        q_neg = Merge(
            mode=lambda x: x[0] - x[1],
            output_shape=(self.params['batch_size'], self.q_length, self.wdim),
            name='q-'
        )([q_embedding, q_pos])
        print('q_pos', q_pos._keras_shape, K.ndim(q_pos))
        print('q_neg', q_neg._keras_shape, K.ndim(q_neg))

        a_pos = Merge(
            mode=lambda x: parallel(*x),
            output_shape=(self.params['batch_size'], self.a_length, self.wdim),
            name='a+',
        )([a_embedding, a_match])
        a_neg = Merge(
            mode=lambda x: x[0] - x[1],
            output_shape=(self.params['batch_size'], self.a_length, self.wdim),
            name='a-'
        )([a_embedding, a_pos])
        print('a_pos', a_pos._keras_shape, K.ndim(a_pos))
        print('a_neg', a_neg._keras_shape, K.ndim(a_neg))
        self.tensors['q+'] = q_pos
        self.tensors['q-'] = q_neg
        self.tensors['a+'] = a_pos
        self.tensors['a-'] = a_neg
   
    def buildConvolution(self, name):
        filters = self.params.get('filters')
        nb_filter = self.params.get('nb_filter')
        assert filters
        assert nb_filter
        convs = []
        for fsz in filters:
            layer_name = '%s-conv-%d' % (name, fsz)
            conv = Convolution1D(
                nb_filter=nb_filter,
                filter_length=fsz,
                border_mode='valid',
                #activation='relu',
                subsample_length=1,
                init='glorot_uniform',
                #init=init,
                #init=lambda shape, name: initializations.uniform(shape, scale=0.01, name=name),
                W_constraint=maxnorm(self.params.get('w_maxnorm')),
                b_constraint=maxnorm(self.params.get('b_maxnorm')),
                #W_regularizer=regularizers.l2(self.params.get('w_l2')),
                #b_regularizer=regularizers.l2(self.params.get('b_l2')),
                #input_shape=(self.q_length, self.wdim),
                name=layer_name
            )
            convs.append(conv)
        self.layers['%s-convolution' % name] = convs

    def linkFeature(self, input_name, conv_name, activation='tanh'):
        filters = self.params.get('filters')
        nb_filter = self.params.get('nb_filter')
        convs = self.layers.get(conv_name)
        assert filters
        assert convs
        features = []
        for fsz, conv in zip(filters, convs):
            conv_output = conv(self.tensors[input_name])
            if type(activation) == type(''):
                act = Activation(
                    activation, name='%s-act-%d' % (input_name, fsz)
                )(conv_output)
            else:
                act = activation(
                    name='%s-advanced-act-%d' % (input_name, fsz)
                )(conv_output)
            maxpool = Lambda(
                lambda x: K.max(x, axis=1),
                output_shape=(nb_filter,),
                name='%s-maxpool-%d' % (input_name, fsz)
            )(act)
            features.append(maxpool)
        if len(features) > 1:
            return Merge(mode='concat', name='%s-feature' % input_name)(features)
        else:
            return features[0]

    def buildFeatures(self, type='shared'):
        assert self.checkTensor('q+')
        assert self.checkTensor('q-')
        assert self.checkTensor('a+')
        assert self.checkTensor('a-')
        srelu = lambda name: SReLU(name=name)
        features = []
        if type == 'shared':
            q_features = Merge(
                mode='concat',
                name='q-features',
            )([
                self.linkFeature('q+', 'shared-convolution', activation=srelu),
                self.linkFeature('q-', 'shared-convolution', activation=srelu)
            ])
            a_features = Merge(
                mode='concat',
                name='a-features',
            )([
                self.linkFeature('a+', 'shared-convolution', activation=srelu),
                self.linkFeature('a-', 'shared-convolution', activation=srelu)
            ])
        else:
            raise Error('Not Supported')
        self.tensors['q-features'] = q_features
        self.tensors['a-features'] = a_features

