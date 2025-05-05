import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
from scipy.fft import fft
from scipy.signal import stft
from numpy import angle, mean, var

def calculate_sim(ri, rj, sim_strategy):
    """
    Calculate similarity between region ri and rj using diverse
    combinations of similarity measures.
    R: correlation, L: length, V: variance.
    """
    sim = 0

    if 'R' in sim_strategy:
        # sim += _calculate_correlation_sim(ri, rj)
        sim += _calculate_cross_correlation(ri, rj)
    # if 'S' in sim_strategy:
    #     sim += _calculate_self_correlation_sim(ri, rj)
    if 'F' in sim_strategy:
        sim += _calculate_frequency_similarity(ri, rj)
    if 'C' in sim_strategy:
        sim += _calculate_correlation_coefficient(ri, rj)
    if 'L' in sim_strategy:
        sim += _calculate_length_sim(ri, rj)
    if 'V' in sim_strategy:
        sim += _calculate_variance_sim(ri, rj)

    return sim

def _pad_signals(ri, rj):
    """
    Pad signals to make them the same length.
    """
    max_len = max(len(ri['data']), len(rj['data']))
    ri_padded = np.pad(ri['data'], (0, max_len - len(ri['data'])), 'constant')
    rj_padded = np.pad(rj['data'], (0, max_len - len(rj['data'])), 'constant')
    return ri_padded, rj_padded

def _calculate_correlation_coefficient(ri, rj):
    """
    Calculate similarity based on the correlation coefficient of I/Q samples.
    """
    ri_padded, rj_padded = _pad_signals(ri, rj)
    real_corr = np.corrcoef(ri_padded.real, rj_padded.real)[0, 1]
    imag_corr = np.corrcoef(ri_padded.imag, rj_padded.imag)[0, 1]
    return (real_corr + imag_corr) / 2

def _calculate_cross_correlation(ri, rj):
    """
    Calculate similarity based on the maximum cross-correlation of I/Q samples.
    """
    ri_padded, rj_padded = _pad_signals(ri, rj)
    cross_corr = correlate(ri_padded, rj_padded, mode='full')
    norm_corr = cross_corr / (np.linalg.norm(ri['data']) * np.linalg.norm(rj['data']))
    return max(abs(norm_corr))

def _calculate_euclidean_distance(ri, rj):
    """
    Calculate similarity based on the Euclidean distance of I/Q samples.
    """
    ri_padded, rj_padded = _pad_signals(ri, rj)
    return euclidean(ri_padded, rj_padded)

# def _calculate_phase_coherence(ri, rj):
#     """
#     Calculate similarity based on the phase coherence of I/Q samples.
#     """
#     ri_padded, rj_padded = _pad_signals(ri, rj)
#     phase_diff = angle(ri_padded) - angle(rj_padded)
#     coherence = mean(abs(np.exp(1j * phase_diff)))
#     return coherence

def _calculate_frequency_similarity(ri, rj):
    """
    Calculate similarity based on the spectral similarity of I/Q samples.
    """
    corrf = np.corrcoef(ri["frequency_hist"], rj["frequency_hist"])[0, 1]

    disi = np.max(ri["frequency_hist"]) - np.min(ri["frequency_hist"])
    disj = np.max(rj["frequency_hist"]) - np.min(rj["frequency_hist"])

    #corrf = sum([max(a, b) for a, b in zip(ri["frequency_hist"], rj["frequency_hist"])])
    return corrf

    #return sum([min(a, b) for a, b in zip(ri["frequency_hist"], rj["frequency_hist"])])

def calculate_frequency_histogram(mask, iq_data, window_size, nperseg=None, noverlap=None):
    if nperseg is None:
        nperseg = window_size
    if noverlap is None:
        noverlap = nperseg // 2

    iq_samples = iq_data[mask]
    # 计算短时傅里叶变换
    _, _, Zxx = stft(iq_samples, nperseg=nperseg, noverlap=noverlap, return_onesided=False)

    # 计算每个频率下的最大值
    max_magnitudes = np.max(np.abs(Zxx), axis=1)

    return max_magnitudes


# def _calculate_correlation_sim(ri, rj):
#     """
#     Calculate similarity using normalized cross-correlation.
#     """
#     signali = ri['data']
#     signalj = rj['data']
#     corr = correlate(signali, signalj)
#     norm_corr = corr / (np.linalg.norm(signali) * np.linalg.norm(signalj))
#
#     # avg_corr = np.abs( np.mean(norm_corr) )
#
#     max_correlation = np.max(np.abs(norm_corr))
#     offset = np.argmax(np.abs(norm_corr)) - (len(signalj) - 1)
#
#     return max_correlation


# def _calculate_self_correlation_sim(ri, rj):
#     self_corri = correlate(ri['data'], ri['data'])
#     norm_self_corri = self_corri / (np.linalg.norm(ri['data'])**2)
#     self_corrj = correlate(rj['data'], rj['data'])
#     norm_self_corrj = self_corrj / (np.linalg.norm(rj['data'])**2)
#
#     return 1

def _calculate_length_sim(ri, rj):
    """
    Calculate similarity based on the length of the I/Q samples.
    """
    length_ri = len(ri['data'])
    length_rj = len(rj['data'])
    max_length = max(length_ri, length_rj)
    return 1 - abs(length_ri - length_rj) / max_length

def _calculate_variance_sim(ri, rj):
    """
    Calculate similarity based on the variance of the I/Q samples.
    """
    #var_ri = np.var(ri['data'])
    #var_rj = np.var(rj['data'])
    #return 1 - abs(var_ri - var_rj) / max(var_ri, var_rj)

    var_ri = np.mean(np.abs(ri['data'])**2)
    var_rj = np.mean(np.abs(rj['data'])**2)
    return 1 - abs(var_ri - var_rj) / max(var_ri, var_rj)