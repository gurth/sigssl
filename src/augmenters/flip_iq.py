import numpy as np

class FlipIQ:
    def __init__(self, random_ratio=0):
        self.random_ratio = random_ratio

    def run(self, data):
        # Ensure the input data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data should be a NumPy array")

        # Ensure the input data has the correct dimensions
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input data should have shape [length, 2]")

        flipped_data = data[:, [1, 0]]

        return flipped_data

    def __call__(self, data):
        return self.run(data)

    def random_apply(self, data):
        if np.random.rand() < self.random_ratio:
            return self.run(data)
        return data