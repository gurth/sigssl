

class BaseProposal():
    def __init__(self, max_prop=-1):
        self.max_prop = max_prop

    def process(self, data):
        raise NotImplementedError()
    def preprocess(self, data):
        raise NotImplementedError()

    def postprocess(self, bboxs):
        raise NotImplementedError()

    def __call__(self, data):
        data = self.preprocess(data)
        bboxs = self.process(data)
        bboxs = self.postprocess(bboxs)

        return bboxs

    @staticmethod
    def merge_bboxs(bboxs, n_smooth=0):
        merged = [bboxs[0]]
        for current in bboxs[1:]:
            last_merged = merged[-1]

            if current[0] <= last_merged[1] + n_smooth:
                last_merged[1] = max(last_merged[1], current[1])
                last_merged[2] = min(last_merged[2], current[2])
            else:
                merged.append(current)

        return merged