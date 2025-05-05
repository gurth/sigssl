import numpy as np


def one_d_felzenszwalb(signal, scale=1, sigma=0.8, min_size=1):
    length = len(signal)

    # 初始每个点作为一个分割区域
    labels = np.arange(length)

    # 计算信号的差值
    diffs = np.abs(np.diff(signal))

    # 按差值排序的边列表
    edges = np.argsort(diffs)

    # 初始化每个区域的大小和内部差值
    region_sizes = np.ones(length)
    internal_diffs = np.zeros(length)

    def merge_regions(label1, label2):
        new_label = min(label1, label2)
        old_label = max(label1, label2)

        region_sizes[new_label] += region_sizes[old_label]
        internal_diffs[new_label] = max(internal_diffs[new_label], diffs[min(old_label, length - 2)])

        labels[labels == old_label] = new_label

    # 按边列表顺序合并区域
    for edge in edges:
        left = edge
        right = edge + 1

        if labels[left] != labels[right]:
            left_label = labels[left]
            right_label = labels[right]

            diff = diffs[edge]

            if diff < scale * internal_diffs[left_label] and diff < scale * internal_diffs[right_label]:
                merge_regions(left_label, right_label)

    # 最小区域大小约束
    for edge in edges:
        left = edge
        right = edge + 1

        if labels[left] != labels[right]:
            left_label = labels[left]
            right_label = labels[right]

            if region_sizes[left_label] < min_size or region_sizes[right_label] < min_size:
                merge_regions(left_label, right_label)

    return labels

def simple_segment(iq_data, k):
    nsplit = int(len(iq_data) / k)
    labels = np.zeros(len(iq_data), dtype=int)

    for i in range(nsplit):
        start = i * k
        end = (i + 1) * k
        labels[start:end] = i

    if len(iq_data) % k > 0:
        labels[nsplit*k:] = nsplit

    return labels

def segment_iq_data(iq_data, k, strategy="felzenszwalb"):
    labels = None
    if strategy == "felzenszwalb":
        labels = one_d_felzenszwalb(iq_data, scale=100, sigma=0.8, min_size=k)
    elif strategy == "simple":
        labels = simple_segment(iq_data, k)
    else:
        raise RuntimeError("Not a valid segment strategy: {}.".format(strategy))

    return labels



