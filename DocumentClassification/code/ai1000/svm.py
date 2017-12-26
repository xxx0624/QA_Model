# -*- coding=utf8 -*-
from sklearn.svm import SVC, NuSVC, SVR
import os, codecs

def load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                    test_emb_data_path, test_tag_data_path, test_label_data_path):
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
                if index % 150 == 0:
                    x_train.append(current_sentence_emb)
                    x_train_2.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / 150, ' x-train(1) size = ', len(x_train)

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
                if index % 150 == 0:
                    x_train[int(index / 150) - 1].extend(current_sentence_tag_emb)
                    current_sentence_tag_emb = []
                print index / 150, ' x-train(2) size = ', len(x_train)

    y_train = []
    with codecs.open(train_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = int(float(line)) - 1
                y_train.append(emb)
    print 'y-train size = ', len(y_train)

    # load test data
    x_test = []
    x_test_2 = []
    with codecs.open(test_emb_data_path, 'r', encoding='utf-8') as fr:
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
                if index % 150 == 0:
                    x_test.append(current_sentence_emb)
                    x_test_2.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / 150, ' x-test(1) size = ', len(x_test)

    with codecs.open(test_tag_data_path, 'r', encoding='utf-8') as fr:
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
                if index % 150 == 0:
                    x_test[int(index / 150) - 1].extend(current_sentence_tag_emb)
                    current_sentence_tag_emb = []
                print index / 150, ' x-test(2) size = ', len(x_test)


    y_test = []
    with codecs.open(test_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = int(float(line)) - 1
                y_test.append(emb)
    print 'y-test size = ', len(y_test)

    return x_train, x_train_2, y_train, x_test, x_test_2, y_test

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



def svc(x_train, y_train, x_test, y_test):
    clf = NuSVC()  # class
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

    train_emb_data_path = os.path.join('Data', 'training.seg.word.emb')
    train_tag_data_path = os.path.join('Data', 'training.seg.tag.emb')
    train_label_data_path = os.path.join('Data', 'training.seg.label.emb')
    test_emb_data_path = os.path.join('Data', 'testing.seg.word.emb')
    test_tag_data_path = os.path.join('Data', 'testing.seg.tag.emb')
    test_label_data_path = os.path.join('Data', 'testing.seg.label.emb')
    x_train, x_train_2, y_train, x_test, x_test_2, y_test = load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                                                   test_emb_data_path, test_tag_data_path, test_label_data_path)
    svm(x_train, y_train, x_test, y_test)
    svr(x_train, y_train, x_test, y_test)
    svc(x_train, y_train, x_test, y_test)

    svm(x_train_2, y_train, x_test_2, y_test)
    svr(x_train_2, y_train, x_test_2, y_test)
    svc(x_train_2, y_train, x_test_2, y_test)