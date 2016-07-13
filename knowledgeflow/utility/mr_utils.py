import numpy as np
import re
import itertools
from collections import Counter
from utility import build_vocab, clean_str

def load_data_and_labels(pos, neg):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(pos).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    #x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_input_data(sentences, labels, word2index):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[word2index[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def mapping_input(sentences, word2index):
    x = np.array([[word2index[word] for word in sentence] for sentence in sentences], dtype='int32')
    return x
    
def hasA_butB(sentences, padding_word="<PAD/>"):
    sentences_S = []
    sentences_B = []
    for sentence in sentences:
        if not 'but' in sentence: continue
        i = sentence.index('but')
        sentences_S.append(sentence[:])
        sentences_B.append([padding_word] * (i + 1) + sentence[i+1:])
    return sentences_S, sentences_B

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, word2index = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, word2index)
    return [x, y, vocabulary, word2index]

if __name__ == '__main__':
    load_data()
