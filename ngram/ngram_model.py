__author__ = 'yeborui'

import operator
import sys
import smoothing as sm
import math

class NgramModel:
    all_count = 0
    word_count = {}
    combine_count = {}
    word_number = 0
    combine_number = 0
    gram = 0

    def load_data(self, file_name, gram=2,
                  word_number=8000, combine_number=5000000):
        print 'loading data from file: ' + file_name
        self.word_number = word_number
        self.combine_number = combine_number
        self.gram = gram
        self.process_file(file_name, gram, word_number, combine_number)
        return

    def test_data(self, file_name, window_size=10):
        print "loading test file: " + file_name
        score_count = 0.0
        sent_count = 0
        f = open(file_name, 'r')
        W_t = 0
        line_count = 0
        for lines in f:
            line_count += 1
            if line_count % 5000 == 0:
                print 'test time:' + str(line_count)
            token = lines.strip().split(' ')
            W_t += len(token)
            i = 0
            while(i < len(token)):
                prob = 1.0
                for j in range(i, i + window_size):
                    if j == len(token):
                        break
                    prob = sm.sentence_addictive_smoothing(
                        token[i:i+window_size],
                        self.word_number,
                        self.word_count,
                        self.combine_count,
                        self.gram,
                        self.all_count
                    )
                i += window_size
                score_count += math.log(prob, 2)
                sent_count += 1
        hp = -score_count / W_t
        print 'perplexity:' + str(math.pow(2, hp))
        return math.pow(2, hp)

    def add_word_count(self, dic, word):
        word = word.encode('utf-8').decode('utf-8')
        if dic.has_key(word):
            dic[word] += 1
        else:
            dic[word] = 1

    def clear_count(self, dic, number):
        dic = sorted(dic.items(),
                     key=operator.itemgetter(1))
        dic.reverse()
        iter = 0
        new_count = {}
        for word,count in dic:
            iter += 1
            if iter > number:
                break
            new_count[word] = count
        dic = new_count
        return dic

    '''
        process the file and store the count results
        @:param file_name: name of file
        @:param gram: n-gram n
        @:param word_number: word number restriction
        @:param combine_number: word combination number restriction
    '''
    def process_file(self, file_name, gram, word_number, combine_number):
        f = open(file_name, 'r')
        '''
            firstly go through the file to count words' ocurrence
            then cut the words with low frequency, resulting in word_number words
        '''
        line_count = 0
        for lines in f:
            token = lines.strip().split(' ')
            line_count += 1
            if line_count % 5000 == 0:
                print 'second time:' + str(line_count)
            for i in range(len(token)):
                self.add_word_count(self.word_count, token[i])
        '''
            cut word count volume and resize size
        '''
        self.word_count = self.clear_count(self.word_count, word_number)
        self.word_number = min(self.word_number, len(self.word_count))
        f.close()
        '''
            traverse the file again to count the number of co-ocurrence
        '''
        line_count = 0
        f = open(file_name, 'r')
        for lines in f:
            line_count += 1
            if line_count % 5000 == 0:
                print 'first time:' + str(line_count)
            token = lines.strip().split(' ')
            self.all_count += len(token)
            for i in range(len(token)):
                if not self.word_count.has_key(
                        token[i].encode('utf-8').decode('utf-8')):
                    continue
                concat_string = token[i].encode('utf-8').decode('utf-8')
                for j in range(1, gram):
                    if i-j < 0:
                        break
                    concat_string = token[i-j].encode('utf-8').decode('utf-8')\
                                    + concat_string
                    self.add_word_count(self.combine_count,
                                        concat_string)
        '''
            shrinking size
        '''
        self.combine_count = self.clear_count(
            self.combine_count, combine_number)
        f.close()
        return

reload(sys)
sys.setdefaultencoding('utf-8')
model = NgramModel()
model.load_data('../data/merge', 3)
model.test_data('../data/merge', 10)
