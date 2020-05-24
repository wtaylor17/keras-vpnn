from vpnn import vpnn
import tensorflow as tf
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255

model = vpnn(28*28,
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

import matplotlib.pyplot as plt

_, axs = plt.subplots(1, 2)
axs[0].plot(history['val_loss'], 'g', label='val loss')
axs[0].legend()
axs[1].plot(history['val_acc'], 'k', label='val acc')
axs[1].legend()
plt.show()