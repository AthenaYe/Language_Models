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
from keras.layers.core import Dense, Activation, Dropout, Masking
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
from keras.models import model_from_json


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

    def __init__(self, word_vec_url, file_name,
                 word_number=40000, sentence_len=10):
        '''
        :param word_vec_url: url of Word2vec service
        :param file_name: training file name
        :param word_number: size of lexicon
        :param sentence_len: maximum sentence length
        :param word_dimension: word vector dimension
        :param n_symbols: size of lexicon, including masking
        :param file_line: record which line currently reading
        :return: None
        '''
        self.word_vec_url = word_vec_url
        self.file_name = file_name
        self.word_number = word_number
        self.word_dimension = -1
        self.sentence_len = sentence_len
        self.file_line = 0
        self.read_line = 0
        f = open(file_name, 'r')
        print 'in init: start to load file:' + file_name
        count = 0
        for lines in f:
            count += 1
            self.file_line += 1
            if count % 5000 == 0:
                print count
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
            # reserve word_numer index to UNKNOWN WORD
            if i + 2 >= word_number: break
        for word in self.word_count:
            res = requests.get(self.word_vec_url + word.decode('utf-8').encode('utf-8'))
            if res.status_code == 200:
                self.word_embedding[word.decode('utf-8').encode('utf-8')] = [float(values) for values in res.text.strip().split(' ')]
                if self.word_dimension == -1:
                    self.word_dimension = len(self.word_embedding[word])
            else:
                self.word_embedding[word] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        self.word_count['UNKNOWN_WORD'] = len(self.word_count) + 1
        self.index_word[len(self.word_count)] = 'UNKNOWN_WORD'
        self.word_number = len(self.word_count)
        print 'UNknown word index' + str(len(self.word_count))
        self.word_embedding['UNKNOWN_WORD'] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        self.word_embedding[''] = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
        # adding one for mask
        self.n_symbols = len(self.word_count) + 1
        self.embedding_weights = np.zeros((self.n_symbols+1, self.word_dimension)) # ?
        for index, word in self.index_word.items():
            embed = []
            print word
            if self.word_embedding.has_key(word):
            #    print self.word_embedding[word]
                embed = self.word_embedding[word]
            else:
                embed = self.word_embedding['UNKNOWN_WORD']
            if len(embed) < self.word_dimension:
                embed = [np.random.uniform(-0.1, 0.1) for i in range(0, self.word_dimension)]
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

    def build_model(self, load_weights=False, weights_file=''):
        # build the model: 2 stacked LSTM
        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(output_dim=self.word_dimension,
                        input_dim=self.n_symbols+1, # oh my goodness is it n_symbols + 1 or n_symbols
                        mask_zero=True, weights=[self.embedding_weights]))
        self.model.add(Masking(mask_value=0,
                               input_shape=(self.sentence_len, self.word_dimension)))
        self.model.add(LSTM(512, return_sequences=True,
                       input_shape=(self.sentence_len, self.word_dimension)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(512, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.n_symbols))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        try:
            if load_weights == True:
                self.model.load_weights(weights_file)
        except:
            print "no such file called" + weights_file + "!!!"
        print('build finished.')

    def move_file_pointer(self):
        if 'read_line' in os.listdir('.'):
            f = open('read_line', 'r')
            prev_line = long(f.readline())
            f.close()
            while True:
                # need not to be exactly the same line
                if self.read_line in range(prev_line-10, prev_line+10):
                    break
                if self.f.tell() == os.fstat(self.f.fileno()).st_size:
                    self.f.close()
                    self.f = open(self.file_name, 'r')
                    self.read_line = 0
                self.f.readline()
                self.read_line += 1
        else:
            print 'no read_line file!'
            return

    def read_file(self, line_size=1000, read_read_line=False):
        if read_read_line == True:
            self.move_file_pointer()
        sentences = []
        next_chars = []
        for i in range(0, line_size):
            if self.f.tell() == os.fstat(self.f.fileno()).st_size:
                self.f.close()
                self.f = open(self.file_name, 'r')
                self.read_line = 0
            line = self.f.readline()
            self.read_line += 1
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
        X = np.zeros((len(sentences), self.sentence_len), dtype=np.int)
        y = np.zeros((len(sentences), self.n_symbols), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                if not self.word_count.has_key(word):
                    word = 'UNKNOWN_WORD'
                X[i, t] = self.word_count[word]
            nxt = next_chars[i]
            if not self.word_count.has_key(next_chars[i]):
                nxt = 'UNKNOWN_WORD'
            y[i, self.word_count[nxt]] = True
        return  X,y

    def generate_single_sentence(self, sentence, gen_length, diversity=1):
        generated = ''
        for i in range(gen_length):
            x = np.zeros((1, self.sentence_len), dtype=np.int)
            for t, word in enumerate(sentence):
                word = word.decode('utf-8').encode('utf-8')
                if not self.word_count.has_key(word):
                    word = 'UNKNOWN_WORD'
                x[0, t] = self.word_count[word]
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            if next_index == 0: # end signal?
                break
            next_word = self.index_word[next_index]
            generated += next_word + ' '
            sentence = sentence[1:]
            sentence.append(next_word)
        return generated

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
            generated += self.generate_single_sentence(
                generated, sentence, gen_length, diversity
            )
            print 'generated sequences:' + generated

    def cal_prob(self, sent):
        x = np.zeros((1, self.sentence_len))
        mul = 1.0
        for t, word in enumerate(sent):
            word = word.decode('utf-8').encode('utf-8')
            if not self.word_count.has_key(word):
                word = 'UNKNOWN_WORD'
            preds = self.model.predict(x, verbose=0)[0][self.word_count[word]]
            mul *= preds
            x[0, t] = self.word_count[word]
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
        return np.argmax(np.random.multinomial(1, a, 1))\

    def save(self, weights_file_name, file_line_name):
        self.model.save_weights(weights_file_name)
        f = open(file_line_name, 'w')
        f.write(str(self.read_line))
        f.close()

