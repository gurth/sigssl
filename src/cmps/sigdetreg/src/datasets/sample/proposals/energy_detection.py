import numpy as np

from .base_proposal import BaseProposal

class EnergyDetection(BaseProposal):
    def __init__(self, max_prop=-1, merge=True):
        self.win_size = 128
        self.nsmooth = 3 * self.win_size

        super(EnergyDetection, self).__init__(max_prop)
        self.merge = merge

        self.data_len = 131072

        self.name = 'BCED'
        self.L = 16
        self.M = 1
        self.gama = 2.7

    def process(self, data):
        bboxs = []
        nsplit = self.data_len // self.win_size

        for i in range(nsplit):
            c_data = data[(self.win_size * i):(self.win_size * (i + 1)), :]
            c_data = c_data[:, 0] + 1j * c_data[:, 1]

            cov = np.zeros((self.L, self.L), dtype=np.complex128)

            for j in range(self.win_size - self.L):
                segment = c_data[j:j + self.L]
                c_cov = np.outer(segment, np.conj(segment))
                # Covariance matrix
                cov += c_cov

            cov = cov / (self.win_size - self.L)
            # eigenvalues and eigenvectors of covariance matrix
            w, _ = np.linalg.eig(cov)
            # maximum eigenvalue
            max_eig = np.max(w)
            # mu
            mu = np.sum(w) / (self.M * self.L)

            # print(max_eig, mu)

            if np.abs(max_eig) > np.abs(self.gama * mu):
                bboxs.append([self.win_size * i, self.win_size * (i + 1), np.abs(max_eig) / np.abs(self.gama * mu) ])
                # print(len_split * i, len_split * (i + 1))

        if len(bboxs) == 0:
            return []
        else:
            bboxs = sorted(bboxs, key=lambda x: -x[2])
            return bboxs

    def preprocess(self, data):
        return data

    def postprocess(self, bboxs):
        if len(bboxs) == 0:
            return []

        if self.merge:
            bboxs = self.merge_bboxs(bboxs, self.n_smooth)

        if self.max_prop > 0:
            bboxs = bboxs[:self.max_prop]

        return bboxs

