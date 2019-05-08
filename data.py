import cv2
import csv
import numpy as np

class Data:
    def __init__(self, batch_size=512):
        self.data_path = './data'
        samples = []
        with open('{0}/driving_log.csv'.format(self.data_path)) as csvfile:
            reader = csv.reader(csvfile)

            # This skips the first row of the CSV file.
            # csvreader.next() also works in Python 2.
            next(reader)

            for line in reader:
                samples.append(line)

        np.random.shuffle(np.array(samples))
        limit = int(len(samples) * 0.7)
        self.train = samples[:limit]
        self.valid = samples[limit:]
        self.batch_size = batch_size

    def train_length(self):
        return len(self.train) * 3

    def valid_length(self):
        return len(self.valid) * 3

    def generator(self, samples):
        num_samples = len(samples)
        batch_size = self.batch_size // 3

        while True:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                gen_x = np.zeros((len(batch_samples)*3, 160, 320, 3))
                gen_y = np.zeros((len(batch_samples)*3, 1))
                for i, batch_sample in enumerate(batch_samples):
                    center_angle = float(batch_sample[3])
                    correction = 0.2

                    # Center Camera
                    file_name = batch_sample[0].split('/')[-1]
                    gen_x[i] = cv2.imread('{0}/IMG/{1}'.format(self.data_path, file_name))
                    gen_y[i] = center_angle

                    # Left Camera
                    file_name = batch_sample[1].split('/')[-1]
                    gen_x[i] = cv2.imread('{0}/IMG/{1}'.format(self.data_path, file_name))
                    gen_y[i] = center_angle + correction

                    # Right Camera
                    file_name = batch_sample[2].split('/')[-1]
                    gen_x[i] = cv2.imread('{0}/IMG/{1}'.format(self.data_path, file_name))
                    gen_y[i] = center_angle - correction

                yield np.array(gen_x), np.array(gen_y)


    def train_generator(self):
        return self.generator(self.train)

    def valid_generator(self):
        return self.generator(self.valid)
