from os.path import join
from os.path import dirname


import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

WEIGHTS_FILEPATH = 'mnist_cnn.hdf5'
MODEL_ARCH_FILEPATH = 'model.json'

nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='valid', dim_ordering='tf'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model saving callback
checkpointer = ModelCheckpoint(filepath=WEIGHTS_FILEPATH, monitor='val_acc', verbose=1, save_best_only=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)

# Train
batch_size = 128
nb_epoch = 100
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
          callbacks=[checkpointer, early_stopping],
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

with open(join(dirname(__file__), '..', 'client','data', MODEL_ARCH_FILEPATH), 'w') as f:
    f.write(model.to_json())
