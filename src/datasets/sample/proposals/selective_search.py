from .base_proposal import BaseProposal
from .ss.core import selective_search, segment_filter

from itertools import product

def load_strategy(mode):

    cfg = {
        "normal": {
            "ks": [256],
            "sims": ["R", "F"],
            "thrd": {"R": (0.3, 40), "F": (0.3, 40)}
        },
    }

    if isinstance(mode, dict):
        cfg['manual'] = mode
        mode = 'manual'

    ks, sims, thrd = cfg[mode]['ks'], cfg[mode]['sims'], cfg[mode]['thrd']

    return product(ks, sims), thrd


class SelectiveSearch(BaseProposal):
    def __init__(self, max_prop):
        super(SelectiveSearch, self).__init__(max_prop)

        self.name = 'SS'
        self.min_size = 512

    def process(self, data):
        iq_data_complex = data[:, 0] + 1j * data[:, 1]
        vault = selective_search(iq_data_complex, load_strategy, mode='normal')
        filtered_proposals = []
        for x, y in vault:
            filtered_proposals += segment_filter(x, min_size=self.min_size, topN=self.max_prop)

        return filtered_proposals
    def preprocess(self, data):
        return data
    def postprocess(self, bboxs):
        return bboxs