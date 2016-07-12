import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Merge, Dense, Lambda, Activation
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D

def attentive(q, a):
	return tf.batch_matmul(q, a, adj_x=False, adj_y=True, name="attention")

q = Input(shape=[None,10], dtype='float32')
a = Input(shape=[None,10], dtype='float32')
qua = Activation('tanh')(
	Merge(
		mode=lambda x: attentive(*x),
		output_shape=lambda x: x[0][0:2] + x[1][2:]
	)([q, a])
)
print qua.get_shape()
q_softmax = Activation('softmax')(Lambda(lambda x: K.max(x, axis=2, keepdims=True))(qua))
print q_softmax.get_shape()
a_softmax = Activation('softmax')(Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qua))
print a_softmax.get_shape()
sess = tf.Session()
print sess.run([q_softmax], {q:np.random.random((3,5,10)), a:np.random.random((3,7,10))})[0].shape
print sess.run([a_softmax], {q:np.random.random((3,5,10)), a:np.random.random((3,7,10))})[0].shape
