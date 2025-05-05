import numpy as np
from scipy.ndimage import find_objects
from .similarity_measures import calculate_sim, calculate_frequency_histogram


class HierarchicalGrouping(object):
    def __init__(self, iq_data, iq_seg, sim_strategy, win_size=512):
        self.iq_data = iq_data
        self.sim_strategy = sim_strategy
        self.iq_seg = iq_seg.copy()
        self.labels = np.unique(self.iq_seg).tolist()
        self.win_size = win_size

    def build_regions(self):
        self.regions = {}
        unique_labels = np.unique(self.iq_seg)
        for label in unique_labels:
            size = np.sum(self.iq_seg == label)
            objects = find_objects(self.iq_seg == label)
            if not objects:
                continue
            region_slice = objects[0]
            start, stop = region_slice[0].start, region_slice[0].stop
            box = (start, stop)

            mask = self.iq_seg == label
            frequency_hist = calculate_frequency_histogram(mask, self.iq_data, self.win_size)

            region_data = self.iq_data[mask]

            self.regions[label] = {
                'size': size,
                'box': box,
                'data': region_data,
                'frequency_hist': frequency_hist
            }

    def build_region_pairs(self):
        self.s = {}
        for i in self.labels:
            neighbors = self._find_neighbors(i)
            for j in neighbors:
                if i < j:
                    self.s[(i, j)] = calculate_sim(self.regions[i], self.regions[j], self.sim_strategy)

    def _find_neighbors(self, label):
        """
        Parameters
        ----------
        label : int
            label of the region
        Returns
        -------
        neighbors : list
            list of labels of neighbors
        """
        # 创建一个布尔数组，标记出给定标签的边界
        boundaries = np.zeros(len(self.iq_seg), dtype=bool)
        boundaries[:-1] = self.iq_seg[:-1] != self.iq_seg[1:]

        # 找到边界点的索引
        boundary_indices = np.where(boundaries)[0]

        neighbors = set()
        for index in boundary_indices:
            if self.iq_seg[index] == label:
                neighbors.add(self.iq_seg[index + 1])
            elif self.iq_seg[index + 1] == label:
                neighbors.add(self.iq_seg[index])

        return list(neighbors)

    def get_highest_similarity(self):
        highest_similarity = sorted(self.s.items(), key=lambda x: x[1])
        return highest_similarity[-1][0]

    def merge_region(self, i, j):
        # generate a unique label and put in the label list
        new_label = max(self.labels) + 1
        self.labels.append(new_label)

        # merge blobs and update blob set
        ri, rj = self.regions[i], self.regions[j]

        new_size = ri['size'] + rj['size']
        new_box = (min(ri['box'][0], rj['box'][0]), max(ri['box'][1], rj['box'][1]))
        new_data = np.concatenate([ri['data'], rj['data']])
        # Max of i and j
        frequency_hist = np.maximum(ri['frequency_hist'], rj['frequency_hist'])
        # frequency_hist = (ri['frequency_hist'] * ri['size'] + rj['frequency_hist'] * rj['size']) / new_size

        value = {
            'box': new_box,
            'size': new_size,
            'data': new_data,
            'frequency_hist': frequency_hist
        }

        self.regions[new_label] = value

        # update segmentation mask
        self.iq_seg[self.iq_seg == i] = new_label
        self.iq_seg[self.iq_seg == j] = new_label

    def remove_similarities(self, i, j):
        # mark keys for region pairs to be removed
        key_to_delete = []
        for key in self.s.keys():
            if (i in key) or (j in key):
                key_to_delete.append(key)

        for key in key_to_delete:
            del self.s[key]

        # remove old labels in label list
        self.labels.remove(i)
        self.labels.remove(j)

    def calculate_similarity_for_new_region(self):
        i = max(self.labels)
        neighbors = self._find_neighbors(i)

        for j in neighbors:
            # i is larger than j, so use (j,i) instead
            self.s[(j, i)] = calculate_sim(self.regions[i], self.regions[j], self.sim_strategy)

    def is_empty(self):
        if not self.s.keys():
            return True

        # if (max(list(self.s.values()))) < 0.1:
        #     return True

        return False