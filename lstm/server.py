__author__ = 'yeborui'

from flask import Flask, request
import sys
import lstm_lmodel
import jieba

reload(sys)
sys.setdefaultencoding('UTF-8')

rnn_model = lstm_lmodel.RNN_language_model("http://localhost:8000/vector?w=",
                               "../data/test_data")
rnn_model.build_model(load_weights=True, weights_file='my_model_weights_stable.h5')

app = Flask(__name__)

@app.route("/")
def welcome():
    return "Hello World!"

@app.route("/chat")
def chat():
    word = request.args.get('w', None)
    print word
    word = word.decode('utf-8').encode('utf-8')
    seed = ' '.join(jieba.cut(word.strip()))
    segmented = seed.strip().split(' ')
    print seed
    print segmented
    res = ''
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        tmp = rnn_model.generate_single_sentence(segmented, 20, diversity=diversity)
        res += '<p>' + tmp + '</p>'
    return res

app.run(debug=False, host='localhost', port=8888)
