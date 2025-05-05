import numpy as np
import scipy

def split_data(data, labs, nsplit):
    if nsplit == 1:
        return [data], [labs]

    data = np.array_split(data, nsplit, axis=0)
    # Annotations, size is nsplit, each element is a list of dicts
    anns = [[] for _ in range(nsplit)]

    for lab in labs:
        # Total time
        ta = lab['property'][0]
        # Sample frequency
        fs = lab['property'][1]
        # Sample index interval
        ti = int(fs * ta) / nsplit
        # Sample time interval
        ts = 1 / fs

        # Frequency interval
        f1 = lab['bbox'][0]
        f2 = lab['bbox'][2]
        # Time interval
        ti1 = np.floor(lab['bbox'][1] / ts)
        ti2 = np.ceil(lab['bbox'][3] / ts)

        idx = int(ti1 / ti)

        new_property = [ta / nsplit, fs, lab['property'][2], lab['property'][3]]
        # If the annotation is in the same split
        if idx == int(ti2 / ti):
            anns[idx].append({'bbox': [f1, (ti1 % ti) * ts, f2, (ti2 % ti) * ts],
                              'category': lab['category'], "property": new_property})
        # If the annotation is in different splits
        else:
            anns[idx].append({'bbox': [f1, (ti1 % ti) * ts, f2, (ti - 1) * ts],
                              'category': lab['category'], "property": new_property})
            for t in range(idx + 1, int(ti2 / ti)):
                anns[t].append({'bbox': [f1, 0, f2, (ti - 1) * ts],
                                'category': lab['category'], "property": new_property})
            if ti2 % ti > 0:
                anns[int(ti2 / ti)].append({'bbox': [f1, 0, f2, (ti2 % ti) * ts],
                                      'category': lab['category'], "property": new_property})

    return data, anns

def read_bin(bin_path):
    # Read IQ sample data from numpy file
    data = np.load(bin_path)

    return data


def read_lab(lab_path):
    anns = np.load(lab_path, allow_pickle=True)

    for ann in anns:
        # Sample frequency
        fs = ann['property'][1]
        # Sample time interval
        ts = 1 / fs

        # Time interval
        ann['bbox'][1] = np.floor(ann['bbox'][1] / ts)
        ann['bbox'][3] = np.ceil(ann['bbox'][3] / ts)

    return anns

def read_mod_lab(bin_path, lab_path):
    '''
    :param bin_path: I/Q data file path
    :param lab_path: Label file path
    :return: I/Q data, annotations
    '''

    # Read IQ sample data from numpy file
    data = read_bin(bin_path)
    anns = read_lab(lab_path)

    return data, anns