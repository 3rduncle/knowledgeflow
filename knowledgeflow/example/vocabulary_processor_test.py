#coding:utf8
import sys
import numpy as np
from tensorflow.contrib import learn

x_text = [line.decode('utf8').rstrip() for line in open(sys.argv[1])]
vocab_processor = learn.preprocessing.VocabularyProcessor(8)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print x
for word in vocab_processor.vocabulary_._reverse_mapping:
	print word
