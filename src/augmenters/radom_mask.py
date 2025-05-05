import numpy as np

class RandomMask:
    def __init__(self, patch_size=128, mask_ratio = 0.1, random_ratio=0):
        self.random_ratio =random_ratio
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio


    def run(self, data, mask_ratio=0.1):
        # Ensure the input data is a NumPy array
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data should be a NumPy array")

        # Ensure the input data has the correct dimensions
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input data should have shape [length, 2]")

        length = data.shape[0]
        num_patches = length // self.patch_size

        masked_data = np.copy(data)  # Create a copy to avoid modifying the original data

        for i in range(num_patches):
            start_idx = i * self.patch_size
            end_idx = start_idx + self.patch_size

            # Randomly decide whether to mask this patch based on self.ratio
            if np.random.rand() < mask_ratio:
                masked_data[start_idx:end_idx, :] = 0  # Mask the patch with zeros

        return masked_data

    def __call__(self, data):
        return self.run(data, mask_ratio=self.mask_ratio)

    def random_apply(self, data):
        if np.random.rand() < self.random_ratio:
            rand_mask_ratio = np.random.rand() * self.mask_ratio
            return self.run(data, mask_ratio=rand_mask_ratio)
        return data