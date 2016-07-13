#coding:utf8
from __future__ import print_function
from collections import Counter
import itertools
import numpy as np
import random
import os
import re
import sys

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


class StreamDataGenerator(object):
	def __init__(self, path, batch, validation = 0.1, seed = 9527):
		self.fin = open(path)
		self.batch = batch
		self.validation = validation
		self.seed = seed
		self.random = random

	def processor(self, process):
		self.processor = process

	def generate(self):
		while not self.eof():
			train = []
			val = []
			for _ in range(self.batch):
				if self.random.random() > self.validation:
					train.append(self.fin.readline().rstrip('\n'))
				else:
					val.append(self.fin.readline().rstrip('\n'))
			print(len(train), len(val))
			x_train, y_train = self.processor(train)
			x_val, y_val = self.processor(val)
			yield {'train':(x_train, y_train), 'val':(x_val, y_val)}

	def reset(self):
		self.fin.seek(0)
		self.random.seed(self.seed)

	def eof(self):
		return self.fin.tell() == os.fstat(self.fin.fileno()).st_size

def selectMaximumProbability(mat):
	row, col = mat.shape
	m = mat.max(axis = 1)
	indices = mat == np.dot(m.reshape((row, 1)), np.ones((1, col)))
	response = np.zeros_like(mat)
	response[indices] = 1.0
	return response

def build_vocab(sentences, start_with=[]):
    '''
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    '''
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    reversed_vocabulary = start_with + [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    start = len(start_with)
    vocabulary = {x: i for i, x in enumerate(reversed_vocabulary)}
    return [reversed_vocabulary, vocabulary]

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_word2vec(coll, reversed_vocabulary):
    import pymongo
    '''
    Load word2vec from local mongodb
    '''
    word2vec = {}
    hit = 0
    miss = 0
    for word in reversed_vocabulary:
        response = coll.find_one({'word':word})
        if response:
            word2vec[word] = np.array(response['vector'])
            hit += 1
        else:
            miss += 1
    print('hit %d miss %d' % (hit, miss), file=sys.stderr)
    return word2vec

def embedding_layer_weights(reversed_vocabulary, word2vec, dim=300):
    embedding_weights = np.array([word2vec.get(w, np.random.uniform(-0.25,0.25,dim)) for w in reversed_vocabulary])
    return [embedding_weights]

def bucket(sentences, size=5):
    buckets = {}
    for sentence in sentences:
        bucket_id = len(sentence) / size + 1
        buckets.setdefault(bucket_id, []).append(sentence)
    return buckets

def embedding_layer_word2vec(weights, word_idx_map):
    response = []
    response.append('%d %d' % weights.shape)
    for word, idx in word_idx_map.items():
        response.append('%s\t%s' % (word, ' '.join(weights[i].tolist())))
    return response

if __name__ == '__main__':
    #client = pymongo.MongoClient()
    #coll = client['word2vec']['en_google']
    reversed_vocab, vocab = build_vocab([['hello', 'world'],['hello', 'python']], start_with=['<PAD>'])
    print(reversed_vocab, vocab)
    #word2vec = load_word2vec(coll, vocab)
    #print(word2vec)
