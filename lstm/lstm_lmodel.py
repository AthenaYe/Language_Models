'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
THEANO_FLAGS=device=gpu,floatX=float32
'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import operator
import math
import timeit
import requests

reload(sys)
sys.setdefaultencoding('utf-8')


# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")


class RNN_language_model():
    word_map = {}
    word_count = {}
    file_line = 0
    index_word = {}
    word_embedding = {}

    def __init__(self):
        return

    def __init__(self, word_vec_url, file_name, word_number=80000, sentence_len=10):
        self.word_vec_url = word_vec_url
        self.file_name = file_name
        self.word_number = word_number
        self.word_dimension = -1
        self.sentence_len = sentence_len
        f = open(file_name, 'r')
        print 'in init: start to load file:' + file_name
        count = 0
        for lines in f:
            count += 1
            if count % 500000 == 0:
                print count
            self.file_line += 1
            lines = lines.strip()
            token = lines.split(' ')
            for tmp in token:
                tmp = tmp.strip().decode('utf-8').encode('utf-8')
                self.add_map(tmp)
        self.word_map = sorted(self.word_map.items(),
                     key=operator.itemgetter(1))
        self.word_map.reverse()
        for i, tmp in enumerate(self.word_map):
            word = tmp[0].decode('utf-8').encode('utf-8')
            # zero is for masking
            self.word_count[word] = i + 1
            self.index_word[i + 1] = word
            if i + 2 >= word_number: break
        for word in self.word_count:
            res = requests.get(self.word_vec_url + word.decode('utf-8').encode('utf-8'))
            if res.status_code == 200:
                self.word_embedding[word.decode('utf-8').encode('utf-8')] = [float(values) for values in res.text.strip().split(' ')]
                if self.word_dimension == -1:
                    self.word_dimension = len(self.word_embedding[word])
            else:
                self.word_embedding[word] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        self.word_count['UNKNOWN_WORD'] = len(self.word_map) + 1 # ?
        self.index_word[len(self.word_map) + 1] = 'UNKNOWN_WORD'
        print 'UNknown word index' + str(i+2)
        self.word_embedding['UNKNOWN_WORD'] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        self.word_embedding[''] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        # adding one for mask
        self.n_symbols = len(self.word_count) + 1
        self.embedding_weights = np.zeros((self.n_symbols+1, self.word_dimension))
        for index, word in self.index_word.items():
            embed = []
            if self.word_embedding.has_key(word):
                print self.word_embedding[word]
                embed = self.word_embedding[word]
            else:
                embed = self.word_embedding['UNKNOWN_WORD']
            self.embedding_weights[index,:] = np.array(embed)
        print 'loading finishied, the original' \
              'word length is %d, %d after cutting' % (len(self.word_map),
                                                       len(self.word_count))
        self.word_number = len(self.word_count)
        del self.word_map
        f.close()
        self.f = open(self.file_name, 'r')

    def add_map(self, word):
        if self.word_map.has_key(word):
            self.word_map[word] += 1
        else:
            self.word_map[word] = 1
        return

    def build_model(self):
        # build the model: 2 stacked LSTM
        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(output_dim=512,
                        input_dim=self.n_symbols + 1,
                        mask_zero=True, weights=[self.embedding_weights]))
        self.model.add(LSTM(512, return_sequences=True,
                       input_shape=(self.sentence_len, self.word_dimension)))
        self.model.add(Dropout(0.2))
    #    self.model.add(LSTM(512, return_sequences=False))
    #    self.model.add(Dropout(0.2))
        self.model.add(Dense(self.word_number))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        print('build finished.')

    def read_file(self, line_size=1000):
        # print('corpus length:', len(text))

        # char_indices = dict((c, i) for i, c in enumerate(chars))
        # indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = self.sentence_len
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, line_size):
            if self.f.tell() == os.fstat(self.f.fileno()).st_size:
                self.f.close()
                self.f = open(self.file_name, 'r')
            line = self.f.readline()
            token = line.strip().split(' ')
            for i in range(0, len(token), self.sentence_len+1):
                # if i+self.sentence_len > len(token)-1:
                #     break
                maxlen = min(len(token)-1, i+self.sentence_len)
                if maxlen > len(token)-1:
                    break
                sentences.append(token[i: maxlen])
                next_chars.append(token[maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), self.sentence_len,
                      len(self.word_count)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.word_count)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                if not self.word_count.has_key(word):
                    word = 'UNKNOWN_WORD'
                X[i, t, self.word_count[word]] = 1
            nxt = next_chars[i]
            if not self.word_count.has_key(next_chars[i]):
                nxt = 'UNKNOWN_WORD'
            y[i, self.word_count[nxt]] = 1
        return  X,y

    def generate(self, test_file, seed_length, gen_length, diversity=1):
        f = open(test_file, 'r')
        for lines in f:
            token = lines.strip().decode('utf-8').encode('utf-8').split(' ')
            generated = ''
            sentence = token[0: seed_length]
            for tmp in sentence:
                generated += tmp + ' '
            print 'diversity:' + str(diversity)
            print('----- Generating with seed: "' + generated + '"')
            for i in range(gen_length):
                x = np.zeros((1, self.sentence_len, self.word_number))
                for t, word in enumerate(sentence):
                    if not self.word_count.has_key(word):
                        word = 'UNKNOWN_WORD'
                    x[0, t, self.word_count[word]] = 1
                preds = self.model.predict(x, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_word = self.index_word[next_index]
                generated += next_word + ' '
                sentence = sentence[1:]
                sentence.append(next_word)
            print 'generated sequences:' + generated

    def cal_prob(self, sent):
        x = np.zeros((1, self.sentence_len, self.word_number))
        mul = 1.0
        for t, word in enumerate(sent):
            word = word.decode('utf-8').encode('utf-8')
            if not self.word_count.has_key(word):
                word = 'UNKNOWN_WORD'
            preds = self.model.predict(x, verbose=0)[0][self.word_count[word]]
            mul *= preds
            x[0, t, self.word_count[word]] = 1
        print 'prob of sent:' + str(mul)
        return mul

    def cal_perplexity(self, test_file):
        W_t = 0.0
        res = 0.0
        f = open(test_file, 'r')
        i = 0
        count = 0
        for line in f:
            count += 1
            if count % 5000 == 0:
                print "test" + str(count)
            token = line.strip().decode('utf-8').encode('utf-8').split(' ')
            for i in range(0, len(token)-1, self.sentence_len):
                tmp = token[i:i+self.sentence_len]
                W_t += len(tmp)
                calprob = self.cal_prob(tmp)
                res += math.log(calprob, 2)
        res *= -1 / W_t
        print res, W_t
        if res > 40:
            print "perplexity very high, entropy is" + str(res)
            return -1
        return math.pow(2, res)

    def sample(self, a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

rnn_model = RNN_language_model("http://localhost:8000/vector?w=", "../data/test")
rnn_model.build_model()
# train the model, output generated text after each iteration
line_size = 1000
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    start = timeit.default_timer()
    for i in range(0, rnn_model.file_line / line_size + 1):
        X,y = rnn_model.read_file(line_size=line_size)
        rnn_model.model.fit(X, y, batch_size=min(128, len(X)), nb_epoch=1)
    rnn_model.f.close()
    rnn_model.f = open(rnn_model.file_name, 'r')
    perp = rnn_model.cal_perplexity('../data/test')
    print 'calculate perp:' + str(perp)
    stop = timeit.default_timer()
    print 'time per iter:  ' + str(stop - start)
    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    #     rnn_model.generate('../data/gen', 5, 20, diversity=diversity)
