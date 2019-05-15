import cv2
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt

class Data:
    def __init__(self, batch_size=512):
        """
        Initializes the data structure and reads the CSV

        :param batch_size:
        """
        self.data_path = '/opt/carnd_p3/data'
        # self.data_path = './data'
        self.batch_size = batch_size
        self.augmenting_by = 2
        self.train = []
        self.valid = []

        self.load_normal_distributed_data()

    def load_normal_distributed_data(self):
        """
        Normal distributes the samples according to the steering angle

        :return:
        """
        df = pd.read_csv('{0}/driving_log.csv'.format(self.data_path),
                         names=['center', 'left', 'right', 'steering',
                                'throttle', 'brake', 'speed'],
                         dtype={'center': np.str, 'left': np.str,
                                'right': np.str, 'steering': np.float64,
                                'throttle': np.float64, 'brake': np.float64,
                                'speed': np.float64}, header=0)

        df['steering'].plot.hist(title='Original steering distribution', bins=100)
        plt.savefig("./output/distribution_original.png")
        plt.gcf().clear()

        # From the plot we see that most of the samples are rects, so let's remove some
        zero_indices = df[df['steering'] == 0].index
        df = df.drop(np.random.choice(zero_indices, size=int(len(zero_indices) * 0.9), replace=False))

        df['steering'].plot.hist(title='Final steering distribution', bins=100)
        plt.savefig("./output/distribution_final.png")
        plt.gcf().clear()

        sample_x = [None]*len(df)*3
        sample_y = [None]*len(df)*3

        i = 0
        for index, row in df.iterrows():
            center_steering = float(row['steering'])

            correction = 0.2
            left_steering = center_steering + correction
            right_steering = center_steering - correction

            center_path = row['center'].split('/')[-1]
            left_path = row['left'].split('/')[-1]
            right_path = row['right'].split('/')[-1]

            sample_x[i*3] = center_path
            sample_y[i*3] = center_steering

            sample_x[i*3+1] = left_path
            sample_y[i*3+1] = left_steering

            sample_x[i*3+2] = right_path
            sample_y[i*3+2] = right_steering

            i += 1

        sample_x, sample_y = sklearn.utils.shuffle(sample_x, sample_y)

        samples = np.column_stack((sample_x, np.array(sample_y).astype(object)))
        limit = int(len(samples) * 0.7)
        self.train = samples[:limit]
        self.valid = samples[limit:]

    def train_length(self):
        """
        Total number of images for training

        :return: int
        """
        return len(self.train) * self.augmenting_by

    def valid_length(self):
        """
        Total number of images for validation

        :return: int
        """
        return len(self.valid) * self.augmenting_by

    def generator(self, samples):
        """
        Generates samples to feed the network

        :param samples:
        :return:
        """

        num_samples = len(samples)
        batch_size = self.batch_size // self.augmenting_by

        while True:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                gen_x = np.zeros((len(batch_samples)*self.augmenting_by, 160, 320, 3))
                gen_y = np.zeros((len(batch_samples)*self.augmenting_by, 1))
                for i, batch_sample in enumerate(batch_samples):
                    img = cv2.imread('{0}/IMG/{1}'.format(self.data_path, batch_sample[0]))
                    gen_x[i*self.augmenting_by] = img # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gen_y[i*self.augmenting_by] = batch_sample[1]

                    gen_x[i*self.augmenting_by+1] = cv2.flip(img, 1)
                    gen_y[i*self.augmenting_by+1] = -batch_sample[1]


                yield sklearn.utils.shuffle(np.array(gen_x), np.array(gen_y))


    def train_generator(self):
        return self.generator(self.train)

    def valid_generator(self):
        return self.generator(self.valid)


if __name__ == '__main__':
    data = Data(batch_size=6)
    g = data.train_generator()
    for x, y in g:
        for i, img in enumerate(x):
            cv2.imwrite("output/{}.jpg".format(i), img)
        print('y:', y)
        break