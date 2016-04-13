'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import operator
import math

reload(sys)
sys.setdefaultencoding('utf-8')

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

class RNN_language_model():
    word_map = {}
    word_count = {}
    file_line = 0

    def __init__(self):
        return

    def __init__(self, file_name, word_number=8000, sentence_len=10):
        self.file_name = file_name
        self.word_number = word_number
        self.sentence_len = sentence_len
        f = open(file_name, 'r')
        print 'in init: start to load file:' + file_name
        for lines in f:
            self.file_line += 1
            lines = lines.strip()
            token = lines.split(' ')
            for tmp in token:
                self.add_map(tmp.strip())
        self.word_map = sorted(self.word_map.items(),
                     key=operator.itemgetter(1))
        self.word_map.reverse()
        for i, tmp in enumerate(self.word_map):
            if i > word_number: break
            self.word_count[tmp] = i
        self.word_count['UNKOWN_WORD'] = i+1
        print 'loading finishied, the original' \
              'word length is %d, %d after cutting' % (len(self.word_map),
                                                       len(self.word_count))
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
        self.model.add(LSTM(512, return_sequences=True,
                       input_shape=(self.sentence_len, self.word_number)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(512, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.word_number))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
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
                self.f.open(self.file_name, 'r')
            line = self.f.readline()
            token = line.strip().split(' ')
            for i in range(0, len(token), self.sentence_len):
                maxlen = math.max(len(token), i+self.sentence_len)
                sentences.append(token[i: maxlen])
                next_chars.append(token[i + maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), maxlen, len(self.word_count)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.word_count)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                if not self.word_count.has_key(word):
                    word = 'UNKNOWN_WORD'
                X[i, t, self.word_count] = 1
            nxt = next_chars[i]
            if not self.word_count.has_key(next_chars[i]):
                nxt = 'UNKNOWN_WORD'
            y[i, self.word_count[nxt]] = 1
        return  X,y

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

rnn_model = RNN_language_model("../data/merge")
rnn_model.build_model()
# train the model, output generated text after each iteration
line_size = 1000
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    for i in range(0, rnn_model.file_line / line_size):
        X,y = rnn_model.read_file(line_size=line_size)
        if len(X) < 128:
            continue
        rnn_model.model.fit(X, y, batch_size=128, nb_epoch=1)

    # start_index = random.randint(0, len(text) - maxlen - 1)
    #
    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    #     print()
    #     print('----- diversity:', diversity)
    #
    #     generated = ''
    #     sentence = text[start_index: start_index + maxlen]
    #     generated += sentence
    #     print('----- Generating with seed: "' + sentence + '"')
    #     sys.stdout.write(generated)
    #
    #     for i in range(400):
    #         x = np.zeros((1, maxlen, len(chars)))
    #         for t, char in enumerate(sentence):
    #             x[0, t, char_indices[char]] = 1.
    #
    #         preds = model.predict(x, verbose=0)[0]
    #         next_index = sample(preds, diversity)
    #         next_char = indices_char[next_index]
    #
    #         generated += next_char
    #         sentence = sentence[1:] + next_char
    #
    #         sys.stdout.write(next_char)
    #         sys.stdout.flush()
    #     print()