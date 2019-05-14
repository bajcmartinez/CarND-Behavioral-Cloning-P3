import cv2
import csv
import numpy as np

class Data:
    def __init__(self, batch_size=512):
        # self.data_path = '/opt/carnd_p3/data/'
        self.data_path = './data'
        samples = []
        with open('{0}/driving_log.csv'.format(self.data_path)) as csvfile:
            reader = csv.reader(csvfile)

            # This skips the first row of the CSV file.
            # csvreader.next() also works in Python 2.
            next(reader)

            for line in reader:
                samples.append(line)

        samples = np.array(samples)
        np.random.shuffle(samples)
        limit = int(len(samples) * 0.7)
        self.train = samples[:limit]
        self.valid = samples[limit:]
        self.batch_size = batch_size
        self.augmenting_by = 6

    def train_length(self):
        return len(self.train) * self.augmenting_by

    def valid_length(self):
        return len(self.valid) * self.augmenting_by

    def generator(self, samples):
        num_samples = len(samples)
        batch_size = self.batch_size // self.augmenting_by

        while True:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                gen_x = np.zeros((len(batch_samples)*3*2, 160, 320, 3))
                gen_y = np.zeros((len(batch_samples)*3*2, 1))
                for i, batch_sample in enumerate(batch_samples):
                    center_angle = float(batch_sample[3])
                    correction_factor = 0.2
                    for c, correction in enumerate([-correction_factor, 0, correction_factor]):
                        file_name = batch_sample[0].split('/')[-1]
                        gen_x[i*self.augmenting_by+c] = cv2.imread('{0}/IMG/{1}'.format(self.data_path, file_name))
                        gen_y[i*self.augmenting_by+c] = center_angle + correction

                        gen_x[i*self.augmenting_by+c+3] = np.fliplr(gen_x[i*self.augmenting_by+c])
                        gen_y[i*self.augmenting_by+c+3] = -(center_angle + correction)


                yield np.array(gen_x), np.array(gen_y)


    def train_generator(self):
        return self.generator(self.train)

    def valid_generator(self):
        return self.generator(self.valid)
