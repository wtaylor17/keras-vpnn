import matplotlib.pyplot as plt
plt.rcParams['keymap.pan'].remove('p')
import numpy as np
from keras.models import load_model
import vpnn


class DigitDrawer:
    def __init__(self, ax_p):
        self.ax = ax_p
        self.digit = np.zeros((28, 28))
        self.digit[0, 0] = 255
        self.im = self.ax.imshow(self.digit)
        self.digit[0, 0] = 0
        self.mouse_down = False
        self.model = load_model('mnist-large.h5',
                                custom_objects=vpnn.custom_objects())
        self.update()

    def update(self):
        self.im.set_data(self.digit)
        self.im.axes.figure.canvas.draw()

    def on_move(self, event):
        if not self.mouse_down or not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        # x,y is the center of a 2x2 square
        corners = [[x, y], [x-1, y], [x-1, y-1], [x, y-1]]
        for xc, yc in corners:
            x = int(round(xc))
            y = int(round(yc))
            if 0 <= x < 28 and 0 <= y < 28:
                self.digit[y, x] = 255
        self.update()

    def on_click(self, event):
        self.mouse_down = True

    def on_release(self, event):
        self.mouse_down = False

    def on_button_press(self, event):
        if event.key == 'p':
            pred = self.model.predict(self.digit.reshape(1, 28*28) / 255)[0]
            amax = np.argmax(pred)
            self.im.axes.set_title(f'Prediction: {amax} ({100*pred[amax]}%)')
            self.update()
        elif event.key == 'c':
            self.digit[:, :] = 0
            self.update()


fig, ax = plt.subplots(1, 1)
drawer = DigitDrawer(ax)
cid = plt.gcf().canvas.mpl_connect('button_press_event', drawer.on_click)
cid2 = plt.gcf().canvas.mpl_connect('button_release_event', drawer.on_release)
cid3 = plt.gcf().canvas.mpl_connect('motion_notify_event', drawer.on_move)
cid4 = plt.gcf().canvas.mpl_connect('key_press_event', drawer.on_button_press)
plt.show()
