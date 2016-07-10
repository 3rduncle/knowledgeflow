import tensorflow as tf
import numpy as np

#example = tf.placeholder(tf.float32, [6,10,10], name='input')
example = tf.Variable(np.random.random(6))
batch = tf.train.batch([example], batch_size=4, num_threads=2,  capacity=1, enqueue_many=True)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
print sess.run([batch])[0]
print sess.run([batch])[0]
print sess.run([batch])[0]
