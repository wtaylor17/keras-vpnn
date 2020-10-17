from vpnn import vpnn, custom_objects
import tensorflow as tf
import keras
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False, default='',
                    help='model file to use')

args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_train = x_train.reshape(-1, 28*28) / 255
x_train = np.round(x_train)
x_test = x_test.reshape(-1, 28*28) / 255
x_test = np.round(x_test)

model_path = args.model
if os.path.isfile(model_path):
    model = keras.models.load_model(model_path, custom_objects=custom_objects())
    train_loss, train_acc = model.evaluate(x_train, y_train,
                                           batch_size=256)
    print(f'Train Loss: {train_loss}')
    print(f'Train Accuracy: {train_acc * 100}%')
    test_loss, test_acc = model.evaluate(x_test, y_test,
                                         batch_size=256)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc * 100}%')
else:
    model = vpnn(28 * 28,
                 n_rotations=3,
                 n_layers=3,
                 out_ac='softmax',
                 activation='cheby',
                 cheby_M=2,
                 out_dim=10)
    model.compile(loss='categorical_crossentropy',
                  metrics=['acc'],
                  optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=10, validation_data=[x_test, y_test],
                        batch_size=256).history
    '''
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(1, 2)
    axs[0].plot(history['val_loss'], 'g', label='val loss')
    axs[0].legend()
    axs[1].plot(history['val_acc'], 'k', label='val acc')
    axs[1].legend()
    plt.show()
    '''
    model.save('mnist-vpnn.h5')
    del model
    model = keras.models.load_model('mnist-vpnn.h5',
                                    custom_objects=custom_objects())
    train_loss, train_acc = model.evaluate(x_train, y_train,
                                           batch_size=256)
    print(f'Train Loss: {train_loss}')
    print(f'Train Accuracy: {train_acc * 100}%')
    test_loss, test_acc = model.evaluate(x_test, y_test,
                                         batch_size=256)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc * 100}%')
