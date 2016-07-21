import numpy as np
import tensorflow as tf

freezed_row = tf.Variable(np.zeros((1,5)))
trainable_rows = tf.Variable(np.random.random((9,5)))
w = tf.concat(0, [freezed_row, trainable_rows])
with tf.Session() as sess:
	print sess.run([w])[0].shape
