import sys, codecs, os, logging, time, math, ConfigParser

reload(sys)
sys.setdefaultencoding('utf-8')

import theano

theano.config.floatX = 'float32'

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.optimizers import SGD, Adam


# get config
def getConfig(section, key):
    config = ConfigParser.ConfigParser()
    path = 'cnn.conf'  # os.path.split(os.path.realpath(__file__))[0] + '/db.conf'
    config.read(path)
    return config.get(section, key)


def GetNowTime():
    return str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))


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
                    x_train.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-train(1) size = ', len(x_train)

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

    y_train = []
    with codecs.open(train_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = [int(float(line)) - 1]
                y_train.append(emb)
    print 'y-train size = ', len(y_train)
    y_train = keras.utils.to_categorical(y_train, num_classes=11)

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
                    x_test.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-test(1) size = ', len(x_test)

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

    y_test = []
    with codecs.open(test_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = [int(float(line)) - 1]
                y_test.append(emb)
    print 'y-test size = ', len(y_test)
    y_test = keras.utils.to_categorical(y_test, num_classes=11)

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
                    x_dev.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / sentence_words_num, ' x-dev(1) size = ', len(x_dev)

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

    y_dev = []
    with codecs.open(dev_label_data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                emb = [int(float(line)) - 1]
                y_dev.append(emb)
    print 'y-dev size = ', len(y_dev)
    y_dev = keras.utils.to_categorical(y_dev, num_classes=11)

    return x_train, y_train, x_test, y_test, x_dev, y_dev


'''
# train data
x_train = np.random.random((1000, 300))
x_train = np.expand_dims(x_train, axis=2)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# test data
x_test = np.random.random((100, 300))
x_test = np.expand_dims(x_test, axis=2)
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# dev data
x_dev = np.random.random((5, 300))
x_dev = np.expand_dims(x_dev, axis=2)
'''
train_emb_data_path = os.path.join('Data', 'cnews.train.emb')
train_tag_data_path = os.path.join('Data', 'cnews.train.tag.emb')
train_label_data_path = os.path.join('Data', 'cnews.train.label')
test_emb_data_path = os.path.join('Data', 'cnews.test.emb')
test_tag_data_path = os.path.join('Data', 'cnews.test.tag.emb')
test_label_data_path = os.path.join('Data', 'cnews.test.label')
dev_emb_data_path = os.path.join('Data', 'cnews.dev.emb')
dev_tag_data_path = os.path.join('Data', 'cnews.dev.tag.emb')
dev_label_data_path = os.path.join('Data', 'cnews.dev.label')
x_train, y_train, x_test, y_test, x_dev, y_dev = load_train_data(train_emb_data_path, train_tag_data_path,
                                                                 train_label_data_path,
                                                                 test_emb_data_path, test_tag_data_path,
                                                                 test_label_data_path,
                                                                 dev_emb_data_path, dev_tag_data_path,
                                                                 dev_label_data_path)

# read config
sentence_words_num = int(getConfig('cnn', 'WORDS_DIM'))
word_vector_dim = int(getConfig('cnn', 'VEC_DIM'))
# end config

model = Sequential()
model.add(Conv1D(500, 3, activation='relu', input_shape=(sentence_words_num, 600)))
model.add(MaxPooling1D(3))
model.add(Dropout(1.0))

model.add(Flatten())
model.add(Dense(11, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# test
model.fit(x_train, y_train, batch_size=128, epochs=50)
score = model.evaluate(x_test, y_test, batch_size=64)
print 'loss=', score[0], ' acc=', score[1]

now_time = GetNowTime()
# logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/' + now_time + '.log',
                    filemode='w')
logging.info(str(score[0]) + ',' + str(score[1]))

# save
model_json = model.to_json()
codecs.open('model/' + now_time + '-model_json', 'w', encoding='utf-8').write(model_json)
model.save_weights('model/' + now_time + '-model_weights.h5')
model.save('model/' + now_time + '-model')

# predict
# result = model.predict(x_dev, batch_size=5, verbose=0)
# print result
