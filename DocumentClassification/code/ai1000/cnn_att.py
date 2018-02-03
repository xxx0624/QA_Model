import sys, codecs, os, logging, time, math

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


def GetNowTime():
    return str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))


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
                    x_train.append(attention_word_vec(current_sentence_emb))
                    current_sentence_emb = []
                print index / 150, ' x-train(1) size = ', len(x_train)

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
                    x_test.append(attention_word_vec(current_sentence_emb))
                    current_sentence_emb = []
                print index / 150, ' x-test(1) size = ', len(x_test)

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
#model.save_weights('model/' + now_time + '-model_weights.h5')
model.save('model/' + now_time + '-model')

# predict
# result = model.predict(x_dev, batch_size=5, verbose=0)
# print result
