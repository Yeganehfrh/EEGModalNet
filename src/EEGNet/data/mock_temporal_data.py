import numpy as np


class MockTemporalData():
    def __init__(self, n_samples=10, n_timepoints=100, n_features=2, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.n_samples = n_samples
        self.n_timepoints = n_timepoints
        self.n_features = n_features
        self.split_point = n_samples - int(n_samples * test_size)

    def generate_sample(self, freq):
        return np.sin(np.linspace(0, freq, self.n_timepoints))

    def generate_freq(self):
        return np.random.randint(10, 20, self.n_features)

    def __call__(self):
        x = np.array([self.generate_sample(self.generate_freq()) for _ in range(self.n_samples)])
        ids = np.arange(self.n_samples)
        return x, ids

    def train_loader(self):
        X, ids = self()
        return {'x': X[:self.split_point],
                'y': ids[:self.split_point] > ids.mean(),
                'id': ids[:self.split_point]}

    def test_loader(self):
        X, ids = self()
        return {'x': X[self.split_point:],
                'y': ids[self.split_point:] > ids.mean(),
                'id': ids[self.split_point:]}