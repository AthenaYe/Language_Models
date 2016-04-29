__author__ = 'yeborui'
import timeit
import lstm_lmodel

def start():
    rnn_model = lstm_lmodel.RNN_language_model("http://localhost:8000/vector?w=",
                               "../data/test")
    rnn_model.build_model(load_weights=True, weights_file='my_model_weights_stable.h5')
    # train the model, output generated text after each iteration
    line_size = 1000
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        start = timeit.default_timer()
        for i in range(0, rnn_model.file_line / line_size + 1):
            X,y = rnn_model.read_file(line_size=line_size, read_file_line=True)
            rnn_model.model.fit(X, y, batch_size=min(128, len(X)), nb_epoch=1)
            rnn_model.save('my_model_weights_dev.h5', 'file_line')
        rnn_model.f.close()
        rnn_model.f = open(rnn_model.file_name, 'r')
        perp = rnn_model.cal_perplexity('../data/test')
        print 'calculate perp:' + str(perp)
        stop = timeit.default_timer()
        print 'time per iter:  ' + str(stop - start)
    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    #     rnn_model.generate('../data/test', 5, 20, diversity=diversity)

start()