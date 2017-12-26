import sys, codecs, os, logging, time, math, ConfigParser

reload(sys)
sys.setdefaultencoding('utf-8')


import numpy as np



# get config
def getConfig(section, key):
    config = ConfigParser.ConfigParser()
    path = '../cnn.conf'  # os.path.split(os.path.realpath(__file__))[0] + '/db.conf'
    config.read(path)
    return config.get(section, key)


def calculate_cos(vec1, vec2):
    # 1, 300
    # elements of vec is int
    up_sum = np.multiply(np.mat(vec1), np.mat(vec2)).sum(axis=1).tolist()[0][0]
    down_sum1 = float(0)
    down_sum2 = float(0)
    index = 0
    for i in range(len(vec1)):
        down_sum1 += vec1[i] * vec1[i]
        down_sum2 += vec2[i] * vec2[i]
    down_sum1 = math.sqrt(down_sum1)
    down_sum2 = math.sqrt(down_sum2)
    return float(up_sum / (down_sum1 * down_sum2))


def attention_word_vec(sentence_vec):
    # param:  sentence_vec shape:(150,300)
    # return: shape(150,600)
    sum_list = []
    for i in range(len(sentence_vec)):
        temp_sum = float(0)
        for j in range(len(sentence_vec)):
            if i != j:
                temp_sum += calculate_cos(sentence_vec[i], sentence_vec[j])
        sum_list.append(temp_sum)
    new_sentence_vec = []
    for i in range(len(sentence_vec)):
        temp_sentence_vec = []
        for j in range(len(sentence_vec[0])):
            temp_sentence_vec.append(sentence_vec[i][j] * sum_list[i])
        new_sentence_vec.append(temp_sentence_vec)
    for i in range(len(sentence_vec)):
        sentence_vec[i].extend(new_sentence_vec[i])
    return sentence_vec


def load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                    test_emb_data_path, test_tag_data_path, test_label_data_path,
                    dev_emb_data_path, dev_tag_data_path, dev_label_data_path):
    sentence_words_num = int(getConfig('cnn', 'WORDS_DIM'))
    # load train data
    x_train = []
    with codecs.open(train_emb_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    x_train.append(attention_word_vec(current_sentence_emb))
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-train(1) size = ', len(x_train)
    '''
    with codecs.open(train_tag_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_tag_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_tag_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    for i in range(sentence_words_num):
                        x_train[int(index / sentence_words_num) - 1][i].extend(current_sentence_tag_emb[i])
                    current_sentence_tag_emb = []
                print index / sentence_words_num, ' x-train(2) size = ', len(x_train)
    x_train = np.array(x_train)
    '''

    # load test data
    x_test = []
    with codecs.open(test_emb_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    x_test.append(attention_word_vec(current_sentence_emb))
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-test(1) size = ', len(x_test)
    '''
    with codecs.open(test_tag_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_tag_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_tag_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    for i in range(sentence_words_num):
                        x_test[int(index / sentence_words_num) - 1][i].extend(current_sentence_tag_emb[i])
                    current_sentence_tag_emb = []
                print index / sentence_words_num, ' x-test(2) size = ', len(x_test)
    x_test = np.array(x_test)
    '''

    # load dev data
    x_dev = []
    with codecs.open(dev_emb_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    x_dev.append(attention_word_vec(current_sentence_emb))
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-dev(1) size = ', len(x_dev)
    '''
    with codecs.open(dev_tag_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_tag_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                new_emb_list = []
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        new_emb_list.append(float(emb))
                index += 1
                current_sentence_tag_emb.append(new_emb_list)
                if index % sentence_words_num == 0:
                    for i in range(sentence_words_num):
                        x_dev[int(index / sentence_words_num) - 1][i].extend(current_sentence_tag_emb[i])
                    current_sentence_tag_emb = []
                print index / sentence_words_num, ' x-dev(2) size = ', len(x_dev)
    x_dev = np.array(x_dev)
    '''


    return x_train, x_test, x_dev



train_emb_data_path = os.path.join('.', 'cnews.train.emb')
train_tag_data_path = os.path.join('.', 'cnews.train.tag.emb')
train_label_data_path = os.path.join('.', 'cnews.train.label')
test_emb_data_path = os.path.join('.', 'cnews.test.emb')
test_tag_data_path = os.path.join('.', 'cnews.test.tag.emb')
test_label_data_path = os.path.join('.', 'cnews.test.label')
dev_emb_data_path = os.path.join('.', 'cnews.val.emb')
dev_tag_data_path = os.path.join('.', 'cnews.val.tag.emb')
dev_label_data_path = os.path.join('.', 'cnews.val.label')
x_train, x_test, x_dev = load_train_data(train_emb_data_path, train_tag_data_path,
                                                                 train_label_data_path,
                                                                 test_emb_data_path, test_tag_data_path,
                                                                 test_label_data_path,
                                                                 dev_emb_data_path, dev_tag_data_path,
                                                                 dev_label_data_path)


fw1 = codecs.open('word-allAtt-600.xtrain', 'w', encoding='utf-8')
fw2 = codecs.open('word-allAtt-600.xtest', 'w', encoding='utf-8')
fw3 = codecs.open('word-allAtt-600.xdev', 'w', encoding='utf-8')
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        if j == 0:
            fw1.write(str(x_train[i][j]))
            fw2.write(str(x_test[i][j]))
            fw3.write(str(x_dev[i][j]))
        else:
            fw1.write(',' + str(x_train[i][j]))
            fw2.write(',' + str(x_test[i][j]))
            fw3.write(',' + str(x_dev[i][j]))
    fw1.write('\n')
    fw2.write('\n')
    fw3.write('\n')
fw1.close()
fw2.close()
fw3.close()