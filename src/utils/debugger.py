import cv2
import os

from .radio import read2spectrogram

# RGB colors for 10 classes
colors_hp = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
             [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
             [170, 255, 0], [0, 255, 85], [0, 255, 170]]

# 10 classes
class_name = [
    '4GFSK', 'GMSK', 'pi/4-DQPSK', 'QPSK', '64QAM',
    'OFDM', '16QAM', 'BPSK', 'OQPSK', '8PSK',
]

def spectrogram2jpg(S, anns, index, out_dir):
    img = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for ann in anns:
        bbox = ann['bbox']
        # img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255
        cv2.rectangle(img,
                      (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])),
                      colors_hp[class_name.index(ann['category'])], 2)

    # out_path = os.path.join(out_dir, "mod_bin/{:05d}s.jpg".format(index))
    out_path = os.path.join(out_dir, "{:05d}.jpg".format(index))

    # cv2.imwrite(out_path, img)
    cv2.imwrite(out_path, img)

def debug_save2jpg(n, out_dir, dbg_dir):
    bin_out_path = os.path.join(out_dir, "mod_bin/{:05d}.npy".format(n))
    lab_out_path = os.path.join(out_dir, "mod_lab/{:05d}.npy".format(n))
    S, anns = read2spectrogram(bin_out_path, lab_out_path)
    spectrogram2jpg(S, anns, n, dbg_dir)
