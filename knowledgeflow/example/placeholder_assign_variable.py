import tensorflow as tf
import numpy as np

example = tf.Variable(np.random.random(6))
batch = tf.train.batch([example], batch_size=3, num_threads=2,  capacity=1, enqueue_many=True)
x = tf.placeholder(tf.float64, [1])
y = x + 1
x.assign(batch)
sess = tf.Session()
coord = tf.train.Coordinator()
init = tf.initialize_all_variables()
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
while not coord.should_stop():
	print sess.run([batch])[0]
coord.request_stop()
coord.join(threads)
