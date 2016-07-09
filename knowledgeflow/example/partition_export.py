import random
import numpy as np
import tensorflow as tf

def step1(x):
	with tf.variable_scope('step1'):
		b = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1.0))
	return x + b

def step2(x, reuse=False):
	with tf.variable_scope('step2', reuse=reuse):
		w = tf.get_variable('weigths', [1], initializer=tf.random_normal_initializer())
	return x * w

x1 = tf.placeholder(tf.float32, shape=[1], name="step0")
x2 = tf.placeholder(tf.float32, shape=[1], name="step0")

with tf.Session() as sess:
	m1 = step1(x1)
	y1 = step2(m1)
	y2 = step2(x2, reuse=True)
	sess.run(tf.initialize_all_variables())
	print sess.run(y1, feed_dict={x1:[1.0]})
	print sess.run(y1, feed_dict={m1:[2.0]})
	print sess.run(y2, feed_dict={x2:[2.0]})
	with tf.variable_scope('step2', reuse=True):
		w = tf.get_variable('weigths', [1])
		print w.eval()
		saver = tf.train.Saver([w], sharded=False)
	from tensorflow_serving.session_bundle import exporter
	model_exporter = exporter.Exporter(saver)
	signature = exporter.classification_signature(
		input_tensor=m1, 
		scores_tensor=y1
	)
	model_exporter.init(
		sess.graph.as_graph_def(),
		default_graph_signature=signature
	)
	model_exporter.export('partition_export_weights', tf.constant(0), sess)
	print 'Export Done'
