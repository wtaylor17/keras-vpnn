from vpnn import vpnn
from vpnn.utils import mnist_generator

generation_fn = mnist_generator()
train_generator = map(lambda data: (data[0].reshape(-1,28*28), data[1]), generation_fn())

model = vpnn(28*28,
             n_rotations=4,
             n_layers=2,
             out_ac='softmax',
             activation=None,
             out_dim=10)
model.compile(loss='categorical_crossentropy',
              metrics=['acc'],
              optimizer='adam')
model.summary()

history = model.fit_generator(train_generator,
                                epochs=10,
                                steps_per_epoch=100).history

import matplotlib.pyplot as plt

_, axs = plt.subplots(1, 2)
axs[0].plot(history['loss'], label='loss')
axs[0].legend()
axs[1].plot(history['acc'], label='acc')
axs[1].legend()
plt.show()
