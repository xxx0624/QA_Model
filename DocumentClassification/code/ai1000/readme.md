# Run

1. seg_data.py
2. train_w2v.py
3. emb_data.py
4. cnn.py


# Please Ignore

train data<br>
x_train = np.random.random((1000, 300))<br>
x_train = np.expand_dims(x_train, axis=2)<br>
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)<br>
test data<br>
x_test = np.random.random((100, 300))<br>
x_test = np.expand_dims(x_test, axis=2)<br>
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)<br>
dev data<br>
x_dev = np.random.random((5, 300))<br>
x_dev = np.expand_dims(x_dev, axis=2)<br>
