import numpy as np


class Noiser:
    def __init__(self, e=0.5, random_ratio=0):
        self.e = e
        self.random_ratio = random_ratio

    def run(self, data, e):
        # Ensure the input data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data should be a NumPy array")

        # Ensure the input data has the correct dimensions
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input data should have shape [length, 2]")

        noise = np.random.normal(scale=e, size=data.shape)
        noisy_data = data + noise
        return noisy_data

    def __call__(self, data):
        return self.run(data, self.e)

    def random_apply(self, data):
        if np.random.rand() < self.random_ratio:
            rand_e = np.random.rand() * self.e
            return self.run(data, rand_e)
        return data
