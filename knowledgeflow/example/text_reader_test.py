import sys
import tensorflow as tf

def read_single_line_example(filename):
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
	reader = tf.TextLineReader()
	line, value = reader.read(filename_queue)
	return line, value

line_t, value_t = read_single_line_example(sys.argv[0])
# 将reader返回的tensor打包成batched tensor，
# 注意如果最后几个样本不够一个batch，
# 最后几个样本将失效。
batch_line_t, batch_value_t = tf.train.batch(
    [line_t, value_t], 
	batch_size = 4,
	num_threads = 2,
    capacity = 2000,
)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
while 1:
	try:
		line, value = sess.run([batch_line_t, batch_value_t])
		print line, value
	except tf.python.framework.errors.OutOfRangeError:
		print 'Done'
		break
