import sys
import tensorflow as tf

def read_single_line_example(filename):
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
	reader = tf.TextLineReader()
	line, value = reader.read(filename_queue)
	return line, value

line_t, value_t = read_single_line_example(sys.argv[0])
# 将reader返回的tensor打包成batched tensor，
# 注意：tf.train.batch本身如果遇到输入的tensor的
# 最后几个样本不够组成一个batch。会从输入tensor的
# 头部重新获取。但是，如果输入的是一个reader tensor，
# 它在end of tensor的时候会抛出一个OutOfRangeError的异常，
# 这将导致最后几个样本失效。
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
