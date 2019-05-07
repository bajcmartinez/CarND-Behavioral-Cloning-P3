from keras import applications
from keras import Sequential
from keras.layers import Cropping2D, Dense, Lambda, Flatten, Dropout

from data import Data

class DriverNet:
    def build(self):
        model = Sequential()

        base_model = applications.ResNet50(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')

        self.model = model

    def train(self, batch_size=512, epochs=10):
        data = Data(batch_size=batch_size)
        train_generator = data.train_generator()
        valid_generator = data.valid_generator()

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=data.train_length() / batch_size,
                                 validation_data=valid_generator,
                                 validation_steps=data.valid_length() / batch_size,
                                 epochs=epochs)

if __name__ == '__main__':
    net = DriverNet()
    net.build()
    net.model.summary()
    net.train()

