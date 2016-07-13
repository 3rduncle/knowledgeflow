#coding:utf8
from __future__ import print_function
def generate_label(segments):
	segments = segments.rstrip('\n')
	words = [word.decode('utf8') for word in segments.split()]
	label = ''
	for word in words:
		if len(word) == 1:
			label += 'S'
		else:
			y = ['M' for i in range(len(word))]
			y[0] = 'B'
			y[-1] = 'E'
			label += ''.join(y)
	return ''.join(words), label

def segment2sentence(segments):
    usentences = []
    labels = []
    for seg in segments:
        usentence, label = generate_label(seg)
        usentences.append(usentence)
        labels.append(label)
    return usentences, labels

def label_embedding(label):
	if label == 'B':
		return [1, 0, 0, 0]
	elif label == 'M':
		return [0, 1, 0, 0]
	elif label == 'E':
		return [0, 0, 1, 0]
	else:
		return [0, 0, 0, 1]

def word_window(line, window = (2, 2)):
	prefix = window[0]
	suffix = window[1]
	size = len(line)
	line = ' ' * prefix + line + ' ' * suffix
	return [line[i: i + prefix + suffix + 1] for i in range(size)]

def LabelEncode(label):
	label.tolist()

def SegmentsCount(path):
	counts = [len(GenerateLabel(line.rstrip('\n'))[0]) for line in open(path)]
	print('TOTAL %d MAX %d MIN %d MEAN %f' % (sum(counts), max(counts), min(counts), sum(counts) * 1.0 / len(counts)))

if __name__ == '__main__':
	segments = '你好  ， 世界'
	setence, label = GenerateLabel(segments)
	print(setence)
	print(label)
	print([LabelEmbedding(y) for y in label])
	for line in WordWindow(u'你好'):
		print('[%s]' % line)
