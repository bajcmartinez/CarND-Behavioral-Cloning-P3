# from keras import applications
from keras import Sequential
from keras.layers import Cropping2D, Dense, Lambda, Flatten, Convolution2D
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

from data import Data

class DriverNet:
    def build(self):
        model = Sequential()

        # base_model = applications.VGG19(weights='imagenet', include_top=False)
        # for layer in base_model.layers:
        #     layer.trainable = False

        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))
        # model.add(base_model)

        # nvidia model
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self, batch_size=512, epochs=10):
        data = Data(batch_size=batch_size)
        train_generator = data.train_generator()
        valid_generator = data.valid_generator()

        history = self.model.fit_generator(train_generator,
                                 steps_per_epoch=data.train_length() / batch_size,
                                 validation_data=valid_generator,
                                 validation_steps=data.valid_length() / batch_size,
                                 epochs=epochs,
                                 verbose=1)

        print("Saving model...")
        self.model.save("./model.h5")
        print("Training")
        print(history.history.keys())
        print('Loss:')
        print(history.history['loss'])
        print('Validation Loss:')
        print(history.history['val_loss'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(history.history['acc'])
        ax.plot(history.history['val_acc'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'test'], loc='upper left')
        fig.savefig('train_evol.png')

if __name__ == '__main__':
    net = DriverNet()
    print("Building network...")
    net.build()
    print("Network Summary...")
    net.model.summary()
    print()
    net.train(batch_size=100, epochs=3)

