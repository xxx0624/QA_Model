import sys, codecs, os, logging

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
from keras.callbacks import EarlyStopping

import time


def GetNowTime():
    return str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))


def load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                    test_emb_data_path, test_tag_data_path, test_label_data_path):
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
                if index % 150 == 0:
                    x_train.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / 150, ' x-train(1) size = ', len(x_train)

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
                if index % 150 == 0:
                    for i in range(150):
                        x_train[int(index / 150) - 1][i].extend(current_sentence_tag_emb[i])
                    current_sentence_tag_emb = []
                print index / 150, ' x-train(2) size = ', len(x_train)
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
                if index % 150 == 0:
                    x_test.append(current_sentence_emb)
                    current_sentence_emb = []
                print index / 150, ' x-test(1) size = ', len(x_test)

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
                if index % 150 == 0:
                    for i in range(150):
                        x_test[int(index / 150) - 1][i].extend(current_sentence_tag_emb[i])
                    current_sentence_tag_emb = []
                print index / 150, ' x-test(2) size = ', len(x_test)
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

    return x_train, y_train, x_test, y_test


train_emb_data_path = os.path.join('Data', 'training.seg.word.emb')
train_tag_data_path = os.path.join('Data', 'training.seg.tag.emb')
train_label_data_path = os.path.join('Data', 'training.seg.label.emb')
test_emb_data_path = os.path.join('Data', 'testing.seg.word.emb')
test_tag_data_path = os.path.join('Data', 'testing.seg.tag.emb')
test_label_data_path = os.path.join('Data', 'testing.seg.label.emb')
x_train, y_train, x_test, y_test = load_train_data(train_emb_data_path, train_tag_data_path, train_label_data_path,
                                                   test_emb_data_path, test_tag_data_path, test_label_data_path)

model = Sequential()
model.add(Conv1D(500, 3, activation='relu', input_shape=(150, 600)))
model.add(MaxPooling1D(3))
model.add(Dropout(1.0))

model.add(Flatten())
model.add(Dense(11, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# test
early_stopping = EarlyStopping(monitor='acc', patience=4, mode='max')
model.fit(x_train, y_train, batch_size=128, epochs=50, callbacks=[early_stopping])
score = model.evaluate(x_test, y_test, batch_size=128)
print 'loss=', score[0], ' acc=', score[1]

now_time = GetNowTime()
#logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/' + now_time + '.log',
                    filemode='w')
logging.info(str(score[0])+','+str(score[1]))

#save
model_json = model.to_json()
codecs.open('model/'+now_time+'-model_json', 'w', encoding='utf-8').write(model_json)
#model.save_weights('model/'+now_time+'-model_weights.h5')
model.save('model/'+now_time+'-model')

# predict
# result = model.predict(x_dev, batch_size=5, verbose=0)
# print result
