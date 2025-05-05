import numpy as np
import scipy

import pywt


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


def read2spectrogram(bin_path, lab_path):
    # Set spectrogram parameters
    nfft = 1024
    fs = 512e3

    # Read IQ sample data from numpy file
    data = np.load(bin_path)
    data = data[:, 0] + 1j * data[:, 1]

    # Get spectrogram plot
    f, t, S = scipy.signal.spectrogram(data, fs=fs, nfft=nfft, scaling='spectrum', return_onesided=False)

    # Transform to fit label
    f = np.concatenate((f[f.size // 2:], f[:f.size // 2]))
    S = np.concatenate((S[S.shape[0] // 2:, :], S[:S.shape[0] // 2, :]), axis=0)

    # Read label
    anns = np.load(lab_path, allow_pickle=True)
    new_anns = []

    for ann in anns:
        bbox = ann['bbox']
        # Sample time interval
        ts = 1 / ann['property'][1]
        # Transform to fit label
        f1 = np.searchsorted(f, bbox[0] * 1e3, side='left')
        f2 = np.searchsorted(f, bbox[2] * 1e3, side='right')

        t1 = np.searchsorted(t, bbox[1], side='left')
        t2 = np.searchsorted(t, bbox[3], side='right')

        new_ann = {'bbox': [f1 if f1 > 0 else 0,
                            t1 if t1 > 0 else 0,
                            f2 if f2 < len(f) else len(f) - 1,
                            t2 if t2 < len(t) else len(t) - 1],
                   'category': ann['category']}
        new_anns.append(new_ann)

    return S, new_anns


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


# Read IQ sample data and label from file
# bin_path: path to binary file
# lab_path: path to label file
# return: data, anns
def read_mod_lab(bin_path, lab_path):
    # Read IQ sample data from numpy file
    data = read_bin(bin_path)
    anns = read_lab(lab_path)

    return data, anns

def merge_bboxs(bboxs):
    merged = [bboxs[0]]
    for current in bboxs[1:]:
        last_merged = merged[-1]

        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)

    return merged

def gaussian_radius(det_size, min_overlap=0.85):
    r1 = (1 - min_overlap) * det_size / (2 * min_overlap)
    r2 = (1 - min_overlap) * det_size / (2)
    r3 = (1 - min_overlap) * det_size / (min_overlap + 1)
    return min(r1, r2, r3)


def gaussian1D(diameter, sigma=1):
    radius = (diameter - 1) // 2
    x = np.arange(-radius, radius + 1, 1)
    h = np.exp(-x ** 2 / (2 * sigma ** 2))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1

    gaussian = gaussian1D(diameter, sigma=diameter / 6)

    x = int(center)
    w = heatmap.shape[0]

    left, right = min(x, radius), min(w - x, radius + 1)

    masked_heatmap = heatmap[x - left:x + right]
    masked_gaussian = gaussian[radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def band_pass_filter(s, low_cutoff, high_cutoff, fs, order=2):
    # Normalized cutoff
    nyquist_freq = 0.5 * fs
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq

    b, a = scipy.signal.butter(order, [low, high], btype='band')

    filtered_sig = scipy.signal.lfilter(b, a, s)

    return filtered_sig


def corr_acc(s, kernel):
    enhanced_signal = scipy.signal.correlate(s, kernel, mode='same')

    return enhanced_signal


# Calculate intersection over union
# l1, h1, l2, h2: [batch_size, 1]
# return: [batch_size, iou]
def iou(l1, h1, l2, h2):
    # Calculate intersection
    inter = np.minimum(h1, h2) - np.maximum(l1, l2)
    # inter[inter < 0] = 0
    inter = np.maximum(inter, 0)

    # Calculate union
    union = np.maximum(h1, h2) - np.minimum(l1, l2)
    union = np.maximum(union, 0)

    # Calculate IoU, safe division
    union_safe = np.where(union == 0, 1e-8, union)
    iou = np.where(inter == 0, 0, inter / union_safe)

    return iou


# Soft non-maximum suppression
# dets: [batch_size, max_obj, 4] (t1, t2, score, category)
#       already sorted by score
# Nt: IoU threshold
# sigma: Gaussian sigma
# thresh: score threshold
# method: 1: linear, 2: gaussian
def soft_nms(dets, Nt=0.3, sigma=0.5, thresh=0.001, method="gaussian"):
    N = dets.shape[1]

    t1 = dets[:, :, 0]
    t2 = dets[:, :, 1]
    scores = dets[:, :, 2]

    for i in range(N):
        for j in range(i + 1, N):
            # Calculate IoU
            IoU = iou(t1[:, i], t2[:, i], t1[:, j], t2[:, j])

            if IoU > Nt:
                if method == "linear":
                    weight = np.ones(IoU.shape)
                    weight[IoU > Nt] = weight[IoU > Nt] - IoU[IoU > Nt]
                    scores[:, j] = scores[:, j] * weight
                elif method == "gaussian":
                    scores[:, j] = scores[:, j] * np.exp(-IoU ** 2 / sigma)
                else:
                    raise ValueError("NMS method must be linear or gaussian.")

    # Remove low score
    scores[scores < thresh] = 0

    return dets


# Wavelet decomposition
def wavelet_decomposition(s, wavelet='db4', level=3):
    flag_1d = True
    # If s is not a 1d array, convert it to id array
    if s.ndim != 1:
        if s.shape[1] == 2:
            s = s[:, 0] + 1j * s[:, 1]
            flag_1d = False
        else:
            raise ValueError("The input signal must be a 1d array or a 2d array with 2 columns.")

    coeffs = pywt.wavedec(s, wavelet, level=level)

    if not flag_1d:
        # Convert back to real 2d array
        coeffs = np.array(coeffs)
        coeffs_real = np.real(coeffs)
        coeffs_imag = np.imag(coeffs)
        coeffs = np.column_stack((coeffs_real[0], coeffs_imag[0], coeffs_real[1], coeffs_imag[1]))

    return coeffs


# Wavelet reconstruction
def wavelet_reconstruction(coeffs, wavelet='db4'):
    s = pywt.waverec(coeffs, wavelet)

    return s
