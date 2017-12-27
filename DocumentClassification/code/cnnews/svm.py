# -*- coding=utf8 -*-
from sklearn.svm import SVC, NuSVC, SVR
import os, codecs, ConfigParser

# get config
def getConfig(section, key):
    config = ConfigParser.ConfigParser()
    path = 'cnn.conf'  # os.path.split(os.path.realpath(__file__))[0] + '/db.conf'
    config.read(path)
    return config.get(section, key)

def load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                    dev_emb_data_path, dev_tag_data_path, dev_label_data_path):
    sentence_words_num = int(getConfig('cnn', 'WORDS_DIM'))
    # load train data
    x_train = []
    x_train_2 = []
    with codecs.open(train_emb_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        current_sentence_emb.append(float(emb))
                index += 1
                if index % sentence_words_num == 0:
                    x_train.append(current_sentence_emb)
                    x_train_2.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-train(1) size = ', len(x_train)

    with codecs.open(train_tag_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_tag_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        current_sentence_tag_emb.append(float(emb))
                index += 1
                current_sentence_tag_emb.append(current_sentence_tag_emb)
                if index % sentence_words_num == 0:
                    x_train[int(index / sentence_words_num) - 1].extend(current_sentence_tag_emb)
                    current_sentence_tag_emb = []
                print index / sentence_words_num, ' x-train(2) size = ', len(x_train)

    y_train = []
    with codecs.open(train_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = int(float(line)) - 1
                y_train.append(emb)
    print 'y-train size = ', len(y_train)


    # load dev data
    x_dev = []
    x_dev_2 = []
    with codecs.open(dev_emb_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        current_sentence_emb.append(float(emb))
                index += 1
                if index % sentence_words_num == 0:
                    x_dev.append(current_sentence_emb)
                    x_dev_2.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-dev(1) size = ', len(x_dev)

    with codecs.open(dev_tag_data_path, 'r', encoding='utf-8') as fr:
        index = 0
        current_sentence_tag_emb = []
        for line in fr:
            line = line.strip()
            if line != '':
                emb_list = line.split(',')
                for emb in emb_list:
                    if (emb != '') and (not emb is None):
                        current_sentence_tag_emb.append(float(emb))
                index += 1
                current_sentence_tag_emb.append(current_sentence_tag_emb)
                if index % sentence_words_num == 0:
                    x_dev[int(index / sentence_words_num) - 1].extend(current_sentence_tag_emb)
                    current_sentence_tag_emb = []
                print index / sentence_words_num, ' x-dev(2) size = ', len(x_dev)

    y_dev = []
    with codecs.open(dev_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = int(float(line)) - 1
                y_dev.append(emb)
    print 'y-dev size = ', len(y_dev)

    return x_train, x_train_2, y_train, x_dev, x_dev_2, y_dev

def svm(x_train, y_train, x_test, y_test):
    clf = SVC()  # class
    clf.fit(x_train, y_train)  # training the svc model
    result = clf.predict(x_test) # predict the target of testing samples
    predict_list = result.tolist()
    cnt_true = 0
    for i in range(len(y_test)):
        if int(predict_list[i]) == int(y_test[i]):
            cnt_true += 1
    print float(cnt_true) / float(len(y_test))



def svr(x_train, y_train, x_test, y_test):
    clf = SVR()  # class
    clf.fit(x_train, y_train)  # training the svc model
    result = clf.predict(x_test) # predict the target of testing samples
    predict_list = result.tolist()
    cnt_true = 0
    for i in range(len(y_test)):
        if int(predict_list[i]) == int(y_test[i]):
            cnt_true += 1
    print float(cnt_true) / float(len(y_test))


if __name__ == '__main__':

    train_emb_data_path = os.path.join('Data', 'cnews.train.emb')
    train_tag_data_path = os.path.join('Data', 'cnews.train.tag.emb')
    train_label_data_path = os.path.join('Data', 'cnews.train.label')
    test_emb_data_path = os.path.join('Data', 'cnews.test.emb')
    test_tag_data_path = os.path.join('Data', 'cnews.test.tag.emb')
    test_label_data_path = os.path.join('Data', 'cnews.test.label')
    dev_emb_data_path = os.path.join('Data', 'cnews.val.emb')
    dev_tag_data_path = os.path.join('Data', 'cnews.val.tag.emb')
    dev_label_data_path = os.path.join('Data', 'cnews.val.label')
    x_train, x_train_2, y_train, x_dev, x_dev_2, y_dev = load_train_data(train_emb_data_path, train_tag_data_path,
                                                                 train_label_data_path,

                                                                 dev_emb_data_path, dev_tag_data_path,
                                                                 dev_label_data_path)


    svm(x_train, y_train, x_dev, y_dev)
    svr(x_train, y_train, x_dev, y_dev)

    svm(x_train_2, y_train, x_dev_2, y_dev)
    svr(x_train_2, y_train, x_dev_2, y_dev)
