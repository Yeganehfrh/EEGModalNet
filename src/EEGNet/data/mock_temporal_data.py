import numpy as np


class MockTimeSeries():
    def __init__(self, n_samples=10, n_timepoints=100, n_features=2, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.n_samples = n_samples
        self.n_timepoints = n_timepoints
        self.n_features = n_features

    def generate_sample(self, freq):
        return np.sin(np.linspace(0, freq, self.n_timepoints))

    def generate_freq(self):
        return np.random.randint(10, 20, self.n_features)

    def train_test_split(self):
        dataset = np.array([self.generate_sample(self.generate_freq()) for _ in range(self.n_samples)])
        n_test = int(self.n_samples * self.test_size)
        n_train = self.n_samples - n_test
        return dataset[:n_train], dataset[n_train:]
