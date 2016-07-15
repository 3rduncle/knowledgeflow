#coding:utf8
import sys
import re
import logging
import random
import itertools

from keras.preprocessing import sequence
from utility import build_vocab
from letor_metrics import ndcg_score
import numpy as np

def extract_sentences(fname):
    sentences = []
    start_r = re.compile('<\d+>')
    end_r = re.compile('</\d+>')
    for line in open(fname):
        line = line.rstrip('\n')
        if start_r.match(line):
            phrase = []
            hit = True
            continue
        elif end_r.match(line):
            sentences.append(' '.join(phrase))
            phrase = []
            continue
        elif not line:
            hit = True
            continue
        else:
            pass
        if hit:
            hit = False
            phrase.append(line)
    return sentences

def generate_neg(question, answer):
    qsize = len(question)
    asize = len(answer)
    assert qsize == asize
    neg_q = []
    neg_a = []
    for i in xrange(qsize):
        while True:
            qindex = random.randint(0, qsize - 1)
            aindex = random.randint(0, asize - 1)
            if qindex != aindex and question[qindex] != question[aindex]:
                break
        neg_q.append(question[qindex])
        neg_a.append(answer[aindex])
    return neg_q, neg_a

def extract_qapair(fname, index=[0,1,2]):
    questions = []
    answers = []
    labels = []
    for line in open(fname):
        line = line.rstrip('\n')
        terms = line.split('\t')
        question = terms[index[0]]
        answer = terms[index[1]]
        label = terms[index[2]]
        questions.append(map(lambda x:x.lower(), question.split()))
        answers.append(map(lambda x:x.lower(), answer.split()))
        labels.append(int(label))
    return questions, answers, labels

def extract_balanced_qapair(fname):
    questions = []
    answers = []
    labels = []
    for line in open(fname):
        line = line.rstrip('\n')
        url, question, answer1, socre1, answer2, score2 = line.split('\t')
        question = map(lambda x:x.lower(), question.split())
        questions.append(question)
        answers.append(map(lambda x:x.lower(), answer1.split()))
        labels.append(1)

        questions.append(question)
        answers.append(map(lambda x:x.lower(), answer2.split()))
        labels.append(0)

    return questions, answers, labels

# data format question \t answer \t label
# total C{2,n}/2 pairwise
def extract_ranked_qapair(fname, shuffle=True):
    entries = {}
    for line in open(fname):
        line = line.rstrip('\n')
        url, question, answer, label = line.split('\t')
        question = map(lambda x:x.lower(), question.split())
        answers.append(map(lambda x:x.lower(), answer1.split()))
        entry = entries.setdefault(url, {})
        entry.setdefault('question', question)

        labels = entry.setdefault('label', {})
        labels.setdefault(label, []).append(answer)

    question_answer_pair = []
    for url, entry in entries.items():
        question = entry['question']
        labels = entry['label']
        keys = labels.keys()
        assert len(keys) > 1
        keys = sorted(keys, reverse=False)
        label_pair = zip(keys, keys[1:])
        for high, low in label_pair:
            for ans1, ans2 in itertools.product(labels[high], labels[low]):
                question_answer_pair.append((question, ans1, ans2))

    if shuffle:
        random.shuffle(question_answer_pair)

    questions = []
    answers = []
    for question, ans1, ans2 in question_answer_pair:
        questions.append(question)
        answers.append(ans1)
        questions.append(question)
        answers.append(ans2)
    return questions, answers, [1] * len(questions)

class QaPairs(object):
    def __init__(self, path, loader=extract_qapair):
        self.xq_data, self.xa_data, self.labels = loader(path)
        self.qmax = len(max(self.xq_data, key=lambda x:len(x)))
        self.amax = len(max(self.xa_data, key=lambda x:len(x)))
        self.questions = {}
        self.pos2neg = {}
        self.neg2pos = {}
        self.makeQuestions()

    def makeQuestions(self):
        for idx, (question, answer, label) in enumerate(zip(self.xq_data, self.xa_data, self.labels)):
            entry = self.questions.setdefault(' '.join(question), {})
            entry.setdefault('idx', []).append(idx)
            entry.setdefault('label', []).append(label)
            entry.setdefault('answer', []).append(' '.join(answer))
        for _, entry in self.questions.items():
            pos, neg = [], []
            for idx, label in zip(entry['idx'], entry['label']):
                if label == 1:
                    pos.append(idx)
                else:
                    neg.append(idx)
            for idx in pos:
                self.pos2neg[idx] = neg
            for idx in neg:
                self.neg2pos[idx] = pos

    def build(self, vocabulary, q_length, a_length):
        self.xq_data = [map(lambda x: vocabulary[x], terms) for terms in self.xq_data]
        self.xa_data = [map(lambda x: vocabulary[x], terms) for terms in self.xa_data]
        self.xq_np = sequence.pad_sequences(self.xq_data, maxlen = q_length)
        self.xa_np = sequence.pad_sequences(self.xa_data, maxlen = a_length)
        self.y_np = np.array(self.labels)
        self.built = True

    def shuffle(self):
        idx = np.arange(self.xq_np.shape[0])
        random.shuffle(idx)
        self.xq_np = self.xq_np[idx]
        self.xa_np = self.xa_np[idx]
        self.y_np = self.y_np[idx] 

    def sampling(self, batch = None):
        assert self.built
        if not batch:
            yield self.xq_np, self.xa_np, self.y_np
            return
        total = self.xq_np.shape[0]
        batches = total / batch + 1
        for i in xrange(batches):
            start = i * batch
            end = (i + 1) * batch
            yield self.xq_np[start:end], self.xa_np[start:end], self.y_np[start:end]
        return

class QaPairsTrain(QaPairs):
    def __init__(self, path, **kvargs):
        super(QaPairsTrain, self).__init__(path, **kvargs)

    def partiteSamples(self):
        self.idx_neg = self.y_np == 0
        self.idx_pos = self.y_np == 1
        self.xq_np_neg, self.xq_np_pos = self.xq_np[self.idx_neg], self.xq_np[self.idx_pos]
        self.xa_np_neg, self.xa_np_pos = self.xa_np[self.idx_neg], self.xa_np[self.idx_pos]
        self.y_np_neg, self.y_np_pos = self.y_np[self.idx_neg], self.y_np[self.idx_pos]
        self.isPartited = True

    def underSampling(self):
        assert self.isPartited
        idx = np.arange(self.xq_np_neg.shape[0])
        idx = np.random.choice(idx, self.xq_np_pos.shape[0])
        xq_epoch = np.concatenate((self.xq_np_pos, self.xq_np_neg[idx]))
        xa_epoch = np.concatenate((self.xa_np_pos, self.xa_np_neg[idx]))
        y_epoch = np.concatenate((self.y_np_pos, self.y_np_neg[idx]))
        return xq_epoch, xa_epoch, y_epoch

    def pairwiseSampling(self, batch = None):
        assert self.isPartited
        candidate = np.arange(self.xq_np.shape[0])[self.idx_pos]
        random.shuffle(candidate)
        posids = []
        negids = []
        for pidx in candidate:
            neg = self.pos2neg[pidx]
            if not neg: continue
            nidx = np.random.choice(neg)
            posids.append(pidx)
            negids.append(nidx)
        pairs = len(posids)
        total = pairs * 2
        qshape = list(self.xq_np.shape)
        ashape = list(self.xa_np.shape)
        qshape[0] = total
        ashape[0] = total
        xq_epoch = np.zeros(qshape)
        xa_epoch = np.zeros(ashape)
        xq_epoch[0::2] = self.xq_np[posids]
        xq_epoch[1::2] = self.xq_np[posids]
        xa_epoch[0::2] = self.xa_np[posids]
        xa_epoch[1::2] = self.xa_np[negids]
        y_epoch = np.array([1,0] * pairs)
        if not batch:
            yield xq_epoch, xa_epoch, y_epoch
            return
        batches = total / batch + 1
        for i in xrange(batches):
            start = i * batch
            end = (i + 1) * batch
            yield  xq_epoch[start:end], xa_epoch[start:end], y_epoch[start:end]
        return
    
class QaPairsTest(QaPairs):
    def __init__(self, path, **kvargv):
        super(QaPairsTest, self).__init__(path, **kvargv)
        self.last_predict = {}
    def label_ranking_average_precision_score(self, predictor, batch_size=50):
        from sklearn.metrics import label_ranking_average_precision_score 
        # 计算predict
        p = []
        for xq_batch, xa_batch, _ in super(QaPairsTest, self).sampling(batch_size):
            delta = predictor(xq_batch, xa_batch)
            p += delta[0].tolist()
        p = np.array(p)
        # 筛选可以用来评估的样本
        # 1. 没有正例无法计算得分
        # 2. 没有负例评分没有意义
        map_record = []
        skip1 = 0
        skip2 = 0
        for question, entry in self.questions.items():
            idx = np.array(entry['idx'])
            if self.y_np[idx].max() == 0:
                skip1 += 1
                continue
            if self.y_np[idx].min() != 0:
                skip2 += 1
                #continue
            score = p[idx].reshape(idx.shape).tolist()
            map = label_ranking_average_precision_score(np.array([entry['label']]), np.array([score]))
            map_record.append(map)
        logging.info('Skip1 %d Skip2 %d' % (skip1, skip2))
        return np.array(map_record).mean()

    def label_ranking_average_precision_score2(self, model, batch_size=50): 
        def label_ranking_average_precision_score(label, score):
            assert len(label) == len(score)
            data = zip(label, score)
            data = sorted(data, key=lambda x:x[1],reverse=True)
            count = 0.0
            values = []
            for i in range(len(data)):
                if data[i][0]:
                    count += 1
                    values.append(count / (i + 1))
            assert len(values)
            return sum(values) / count, values[0]
        p = model.predict(
            {'q_input': self.xq_np, 'a_input':self.xa_np},
            batch_size=batch_size
        )
        map_record = []
        for question, entry in self.questions.items():
            idx = np.array(entry['idx'])
            if self.y_np[idx].max() == 0:
                continue
            score = p[idx].reshape(idx.shape).tolist()
            map, _ = label_ranking_average_precision_score(entry['label'], score)
            map_record.append(map)
            self.saveResult(question, map, score)
        map = np.array(map_record).mean()
        self.saveResult('__TOTAL_MAP__', map)
        return map

    def ndcg_score(self, model, k=10, batch_size=50):
        p = model.predict(
            {'q_input': self.xq_np, 'a_input':self.xa_np},
            batch_size=batch_size
        )
        records = []
        for question, entry in self.questions.items():
            idx = np.array(entry['idx'])
            if self.y_np[idx].max() == 0:
                continue
            score = p[idx].reshape(idx.shape).tolist()
            record = ndcg_score(entry['label'], score, k=k)
            records.append(record)
            self.saveResult(question, record, score)
        result = np.array(records).mean()
        self.saveResult('__TOTAL_RESULT__', result)
        return result

    def saveResult(self, question, map, score=None):
        entry = self.last_predict.setdefault(question, {})
        entry['map'] = map
        if score:
            entry['score'] = score

    def dumpResult(self, path):
        with open(path, 'w') as f:
            entry = self.last_predict['__TOTAL_MAP__']
            print >>f, '%s\tNULL\t%f' % ('__TOTAL_MAP__', entry['map'])
            for question, entry in self.questions.items():
                answers = entry['answer']
                predict = self.last_predict.get(question)
                if not predict:
                    continue
                for answer, label, score in zip(answers, entry['label'], predict['score']):
                    print >>f,'%s\t%s\t%d\t%f' % (question, answer, label, score)

if __name__ == '__main__':
    a = extract_sentence('./data/qg/train.answer')
    b = extract_sentence('./data/qg/train.question')
    c, d = generate_neg(a, b)
    print len(a), len(b), len(c), len(d)
        

