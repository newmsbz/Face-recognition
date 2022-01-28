import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
from os import listdir
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback, EarlyStopping

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print_weight = LambdaCallback(on_epoch_end=lambda epoch, logs: print('\n', model.layers[0].get_weights()))
# early_stopping = EarlyStopping(patience=15, mode='auto', monitor='val_loss')

# print_weight = GetWeights()

input_size = 128
# data_limit = 100000


def get_dataset(path):
    x, y = list(), list()

    # Kaggle dataset
    for name in listdir(path + '/same'):
        print(name)     # 165000개
        x.append(np.load(path + '/same/' + name))
        y.append(np.array([0, 1]))  # 같은 사람일 확률
    for name in listdir(path + '/different'):
        print(name)     # 328000개
        x.append(np.load(path + '/different/' + name))
        y.append(np.array([1, 0]))  # 다른 사람일 확률

    return np.array(x), np.array(y)


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(32, input_shape=(128,)),
        # keras.layers.Dense(2, activation='softmax')
        keras.layers.BatchNormalization(),
        keras.layers.Activation(activation='relu'),
        keras.layers.Dropout(rate=0.2),
        # keras.layers.Dense(64),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation(activation='relu'),
        # keras.layers.Dropout(rate=0.2),
        # keras.layers.Dense(32),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation(activation='relu'),
        # keras.layers.Dropout(rate=0.2),
        # keras.layers.Dense(16),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation(activation='relu'),
        # keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(2, activation='softmax'),
    ])

    # model.compile(optimizer='SGD',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    # adam = optimizer.Adam(lr = 0.001)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    dataset_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_train/preprocessed data 2'
    # dataset_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_train/preprocessed data 3'
    # dataset_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_test/preprocessed data 2'
    x, y = get_dataset(dataset_path)
    print(len(x), len(y))
    x = x.reshape((len(x), 128))
    # y = y.reshape((len(y), 128))
    # y = to_categorical(y)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=101)
    print(x_train.shape)
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    print(x.shape, y.shape)
    print(x_train[0].shape)

    model = create_model()
    hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[print_weight])
    # for layer in model.layers: print(layer.get_config(), layer.get_weights())
    # print(model.get_weights())
    # print(model.layers.get_weights())
    for layer in model.layers:
        print(layer)
        for get_weight in layer.get_weights():
            print(get_weight.shape)
            # print(get_weight)


    model.summary()

    plt.figure(figsize=(12, 8))
    plt.title('Loss result')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()

    plt.figure(figsize=(12, 8))
    plt.title('Acc result')
    plt.plot(hist.history['acc'], 'r')
    plt.plot(hist.history['val_acc'], 'g')
    plt.legend(['acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid()
    plt.show()

    model.save(f'test.h5')


