import numpy as np


class FrequencyOffset:
    def __init__(self, frequency_offset=500, sample_rate=512e3, freq_offset_ratio=0, random_ratio=0):
        """
        Initialize the class with frequency offset and sample rate.

        :param frequency_offset: Frequency offset value (in Hz)
        :param sample_rate: Sample rate (in Hz)
        """
        self.frequency_offset = frequency_offset
        self.sample_rate = sample_rate
        self.freq_offset_ratio = freq_offset_ratio

        self.random_ratio = random_ratio

    def run(self, data, frequency_offset=500, sample_rate=512e3):
        # Ensure the input data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data should be a NumPy array")

        # Ensure the input data has the correct dimensions
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input data should have shape [length, 2]")

        # Get the length of the data
        length = data.shape[0]

        # Generate time vector
        t = np.arange(length) / sample_rate

        # Generate frequency offset factor
        freq_offset_factor = np.exp(2j * np.pi * frequency_offset * t)

        # Combine I and Q data into complex form
        complex_data = data[:, 0] + 1j * data[:, 1]

        # Apply frequency offset
        offset_data = complex_data * freq_offset_factor

        # Split I and Q data
        data_with_offset = np.column_stack((offset_data.real, offset_data.imag))

        return data_with_offset

    def __call__(self, data):
        return self.run(data, frequency_offset=self.frequency_offset, sample_rate=self.sample_rate)

    def random_apply(self, data):
        if np.random.rand() < self.random_ratio:
            rand_freq_offset = int(np.random.randn() * self.frequency_offset)
            return self.run(data, frequency_offset=rand_freq_offset, sample_rate=self.sample_rate)
        return data


