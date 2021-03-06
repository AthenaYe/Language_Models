__author__ = 'yeborui'

def addcitive_smoothing(i, token, V, dict, combine_dict, gram, all_count):
    if i == 0:
        if dict.has_key(token[0].encode('utf-8').decode('utf-8')):
        #    print 'has key' + token[0].encode('utf-8').decode('utf-8')
            return dict[token[0].encode('utf-8').decode('utf-8')] * 1.0 / all_count
        else:
            return 1.0 / V
    else:
        j = i - 1
        concat = ''
        while True:
            if j < 0 or i - j >= gram:
                break
            concat = token[j].encode('utf-8').decode('utf-8') + concat
            j -= 1
        num1 = 0
        if combine_dict.has_key(concat.encode('utf-8').decode('utf-8')):
            num1 = combine_dict[concat.encode('utf-8').decode('utf-8')]
        elif dict.has_key(concat.encode('utf-8').decode('utf-8')):
            num1 = dict[concat.encode('utf-8').decode('utf-8')]
        num2 = 0
        concat += token[i].encode('utf-8').decode('utf-8')
        if combine_dict.has_key(concat):
            num2 = combine_dict[concat]

    return (1.0 + num2) / (1 * V + num1)
    #    return (num1 + 1) * 1.0 / (num2 + 15)

def sentence_addictive_smoothing(token, V, dict, combine_dict, gram, all_count):
    prob = 1
    for i in range(len(token)):
        prob *= addcitive_smoothing(i, token, V, dict, combine_dict, gram, all_count)
    return prob
