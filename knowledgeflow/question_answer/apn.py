#coding:utf8
import sys
import json
import struct
import logging
import warnings
import ConfigParser
import numpy as np
import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.initializations import glorot_normal, identity

EPSILON = tf.constant(1e-20)

def margin_hinge(y_true, y_pred, margin=0.5):
	# y_pred are the dot product similarities, in interleaved form (positive example, negative example, ...)
	# y_true is simply 1, 0, 1, 0
	signed = 2 * y_pred * (tf.cast(y_true, tf.float32) - 0.5) # we do this, just so that y_true is part of the computational graph
	batch = tf.shape(signed)[0]
	range1 = tf.range(0, batch, 2)
	range2 = tf.range(1, batch, 2)
	pos = tf.gather(signed, range1)
	neg = tf.gather(signed, range2)
	# negative samples are multiplied by -1, so that the sign in the rankSVM objective is flipped below
	# rank_hinge_loss = K.mean(K.relu(margin - pos - neg))
	rank_hinge_loss = tf.reduce_mean(tf.nn.relu(margin - pos - neg))
	return rank_hinge_loss

class APNBase(object):
	def __init__(self, wdim=300):
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
		with tf.name_scope("input"):
			# Question Network Input
			q_input = tf.placeholder(tf.int32, (None, None), name="q_input")
			# Answer Network Input
			a_input = tf.placeholder(tf.int32, (None, None), name="a_input")
		self.tensors['q_input'] = q_input
		self.tensors['a_input'] = a_input
		
	def setEmbedding(self, weights):
		self.embedding_params['weights'] = weights

	def buildEmbedding(self):
		weights = self.embedding_params.get('weights')
		#assert weights
		trainable = self.params.get('embedding_trainable', False)
		if trainable:
			logging.info('Embedding Weights is Trainable!')
		else:
			logging.info('Embedding Weights is Not Trainable!')
		with tf.name_scope('embedding'):
			W = tf.Variable(
				weights,
				name = 'embedding',
				trainable = trainable,
				dtype = tf.float32
			)
			self.tensors['q_embedding'] = tf.nn.embedding_lookup(W, self.tensors['q_input'])
			self.tensors['a_embedding'] = tf.nn.embedding_lookup(W, self.tensors['a_input'])

	def buildConvolution(self):
		q_embedding = self.tensors['q_embedding']
		a_embedding = self.tensors['a_embedding']
		with tf.name_scope('convolution'):
			filter_shape = (self.params['filters'][0], self.wdim, 1, self.params['nb_filter'])
			W = glorot_normal(filter_shape, name="W")
			b = tf.Variable(tf.constant(0.0, shape=(self.params['nb_filter'],)), name="b")
			q_conv = tf.nn.conv2d(
				tf.expand_dims(q_embedding, -1),
				W,
				strides=[1,1,1,1],
				padding="VALID",
				name="q_conv"
			)
			a_conv = tf.nn.conv2d(
				tf.expand_dims(a_embedding, -1),
				W,
				strides=[1,1,1,1],
				padding="VALID",
				name = "a_conv"
			)
			q_conv = tf.squeeze(q_conv, [2])
			a_conv = tf.squeeze(a_conv, [2])
			# shape = (batch, q_length, NUM_FILTERS)
			q_relu = tf.nn.relu(tf.nn.bias_add(q_conv, b), name="q_relu")
			# shape = (batch, a_length, NUM_FILTERS)
			a_relu = tf.nn.relu(tf.nn.bias_add(a_conv, b), name="q_relu")
		self.tensors['q_conv'] = q_conv
		self.tensors['a_conv'] = a_conv
		self.tensors['q_relu'] = q_relu
		self.tensors['a_relu'] = a_relu
		self.tensors.setdefault('weights', []).append(b)
		self.tensors.setdefault('summary', []).append(tf.nn.zero_fraction(a_relu))

	def buildConvolutionKeras(self):
		q_embedding = self.tensors['q_embedding']
		a_embedding = self.tensors['a_embedding']
		with tf.name_scope('convolution'):
			convolution = Convolution1D(
				nb_filter = self.params['nb_filter'],
				filter_length = self.params['filters'][0],
				border_mode = 'valid',
				activation = 'relu',
				subsample_length = 1,
				init = 'glorot_uniform'
			)
			# shape = (batch, q_length, NUM_FILTERS)
			q_relu = convolution(q_embedding)
			# shape = (batch, a_length, NUM_FILTERS)
			a_relu = convolution(a_embedding)
			self.layers['convolution'] = convolution
			self.tensors['q_relu'] = q_relu
			self.tensors['a_relu'] = a_relu
			self.tensors.setdefault('summary', []).append(tf.nn.zero_fraction(a_relu))

	def buildAttention(self):
		q_relu = self.tensors['q_relu']
		a_relu = self.tensors['a_relu']
		with tf.name_scope("attention"):
			W = identity([self.params['nb_filter'], self.params['nb_filter']], name='W')
			batch = tf.shape(q_relu)[0]
			q_matmul = tf.batch_matmul(q_relu, tf.tile(tf.expand_dims(W,[0]), tf.pack([batch, tf.constant(1), tf.constant(1)])))
			qa_attention = tf.batch_matmul(q_matmul, a_relu, adj_x=False, adj_y=True, name="attention")
			# shape = (batch, q_length, 1)
			qa_attention = tf.tanh(qa_attention)
			q_max = tf.reduce_max(qa_attention, reduction_indices=[2], keep_dims=True, name='q_max')
			# shape = (batch, 1, a_length)
			a_max = tf.reduce_max(qa_attention, reduction_indices=[1], keep_dims=True, name='a_max')
			# shape = (batch, q_length, 1)
			q_softmax = tf.expand_dims(tf.nn.softmax(tf.squeeze(q_max, [2])), -1)
			# shape = (batch, a_length, 1)
			a_softmax = tf.expand_dims(tf.nn.softmax(tf.squeeze(a_max, [1])), -1)
			# https://www.tensorflow.org/versions/r0.9/api_docs/python/math_ops.html#batch_matmul 
			# shape = (batch, NUM_FILTERS, 1)
			q_feature = tf.batch_matmul(q_relu, q_softmax, adj_x=True, adj_y=False)
			a_feature = tf.batch_matmul(a_relu, a_softmax, adj_x=True, adj_y=False)
		self.tensors['q_feature'] = q_feature
		self.tensors['a_feature'] = a_feature
		self.tensors.setdefault('weights', []).append(W)

	def buildSimilarity(self):
		q_feature = self.tensors['q_feature']
		a_feature = self.tensors['a_feature']
		with tf.name_scope('similarity'):
			q_norm = tf.sqrt(tf.reduce_sum(q_feature ** 2, reduction_indices=[1], keep_dims=True))
			a_norm = tf.sqrt(tf.reduce_sum(a_feature ** 2, reduction_indices=[1], keep_dims=True))
			product = tf.batch_matmul(q_feature, a_feature, adj_x=True, adj_y=False, name="product")
			denominator = tf.batch_matmul(q_norm, a_norm, adj_x=False, adj_y=True, name="denominator")
			similarity = tf.squeeze(product / (denominator + EPSILON), [-1,-2], name='similarity')
		self.tensors['similarity'] = similarity

	def buildLoss(self):
		with tf.name_scope("loss"):
			label = tf.placeholder(tf.int32, (None,), name="label")
			score = self.tensors['similarity']
			self.tensors['label'] = label
			self.tensors['loss'] = margin_hinge(label, score, 0.5)

	def build(self):
		self.buildInput()
		self.buildEmbedding()
		self.buildConvolution()
		self.buildAttention()
		self.buildSimilarity()
		self.buildLoss()

	def buildModel(self, input, output):
		pass

	def export(self, path, session):
		saver = tf.train.Saver(self.tensors['weights'], sharded=False)
		from tensorflow_serving.session_bundle import exporter, session_bundle
		model_exporter = exporter.Exporter(saver)
		named_tensor_bindings = {
			"input_q": self.tensors['q_conv'],
			"input_a": self.tensors['a_conv'],
			"output_y": self.tensors['similarity']
		}
		signature = {"generic": exporter.generic_signature(named_tensor_bindings)}
		model_exporter.init(
			session.graph.as_graph_def(),
			named_graph_signatures=signature
		)
		model_exporter.export(path, tf.constant(0), session) 

if __name__ == '__main__':
	from tensorflow_serving.session_bundle import exporter, session_bundle
	conf = ConfigParser.ConfigParser()
	conf.read('./knowledgeflow/question_answer/configure/comment_rank.conf')
	weights = np.random.random((50000,300))
	impl = APNBase(300)
	impl.readDefaultConfig(conf)
	impl.setEmbedding(weights)
	config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=config)
	with sess.as_default():
		tf.set_random_seed(1337)
		print sess.run(tf.random_uniform([1]))
		impl.build()
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer()
		grads_and_vars = optimizer.compute_gradients(impl.tensors['loss'])
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		sess.run(tf.initialize_all_variables())
		print sess.run(tf.random_uniform([1]))
		impl.export('test', sess)
		session_bundle.LoadSessionBundleFromPath('test/00000000')

