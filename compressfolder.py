from compressor.EntropyReduction import EntropyReduction
from jpeg.utils import *
from jpeg.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim
import imageio

from tqdm import tqdm
import argparse

from multiprocessing import Pool

import array
import time
import os


TABLE, QUALITY, SAMPLE_RATIO = QUANTIZATIONTABLE, 64, 1.0


def jpeg(partition):
    return capture(zig_zag(quantize(dct_2d(partition), table=TABLE)), values=QUALITY, sample_percentage=SAMPLE_RATIO)


def SSIM(patch, table=QUANTIZATIONTABLE):
    assert patch.shape[0] % 8 == 0 and patch.shape[1] % 8 == 0, \
        "Invalid sampling area make sure sample area is equally divisible by 8"
    pbar = range(2, 64)
    last_metric, rep = 0, 0
    for i in pbar:
        compressed_data, partitions = array.array('b', []), []
        ext = compressed_data.extend; app = partitions.append
        list_of_patches = split((patch.copy() - 128).astype(np.int8), 8, 8)
        [ext(capture(zig_zag(quantize(dct_2d(x), table=table)), values=i)) for x in list_of_patches]
        compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
        [app(idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)), table=table)) + 128) for y in compressed_split]
        index = merge_blocks(partitions, int(1), int(1)).astype(np.uint8)
        metric = ssim(patch, index, data_range=index.max() - index.min())
        if metric > 0.96:
            if table[0][0] < 8: table[0][0] = 8
            if i % 2 != 0: i += 1
            return i, metric, table
        if abs(last_metric - metric) < 0.0000000001:
            if metric > 0.94:
                if table[0][0] < 8: table[0][0] = 8
                if i % 2 != 0: i += 1
                return i - rep, metric, table
            return SSIM(patch, table=np.round(table/1.1))
        rep += 1
        if rep == 4: last_metric = metric; rep = 0
    if metric < 0.92:
        return SSIM(patch, table=np.round(table/1.2))
    if table[0][0] < 8: table[0][0] = 8
    return 64, metric, table


def precompression_factors(image):
    sample = image[:,:,0]
    c_length, c_width = sample.shape
    p_length, p_width = calc_matrix_eight_size(sample)
    return c_length, c_width, p_length, p_width


def convert_sample(img, length, width):
    YCBCR = rgb2ycbcr(img)
    y, cb, cr = (YCBCR[:, :, 0])[:length, :width], (YCBCR[:, :, 1])[:length, :width], (YCBCR[:, :, 2])[:length, :width]
    return y.astype(np.int16), cb.astype(np.int16), cr.astype(np.int16)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="image folder")
    ap.add_argument("-c", "--compressed", required=True,
                    help="compressed folder")
    args = ap.parse_args()
    start_time = time.time()
    pbar = tqdm(os.listdir(args.images), desc="running compressor on all images in folder")
    for file in pbar:
        filename = os.fsdecode(file)
        resulting_file = args.compressed + os.path.splitext(filename)[0]
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = imageio.imread(args.images + filename)
            length, width = image[:, :, 0].shape; c_l, c_w, p_l, p_w = precompression_factors(image)
            y, cb, cr = convert_sample(image, length, width)
            test_y = matrix_multiple_of_eight(y)
            splits = np.array([np.array(test_y[x:x + 8, z:z + 8], dtype=np.int16) for x in range(0, test_y.shape[0], 8)
                               for z in range(0, test_y.shape[1], 8)], dtype=np.int16)
            variances = [np.var(i) for i in splits]
            original_sample = splits[np.argmax(variances)]
            values_to_keep, metric, quant = SSIM(original_sample, table=QUANTIZATIONTABLE)
            keep = [values_to_keep]; padding = [p_l - c_l, p_w - c_w]
            dimensions = convertBin(p_l, bits=16) + convertBin(p_w, bits=16)
            p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
            p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]

            Y = split((matrix_multiple_of_eight(y - 128)).astype(np.int8), 8, 8)
            CB = split((matrix_multiple_of_eight(cb - 128)).astype(np.int8), 8, 8)
            CR = split((matrix_multiple_of_eight(cr - 128)).astype(np.int8), 8, 8)

            QUALITY, TABLE = values_to_keep, quant
            with Pool(8) as p: compressed_y = array.array('b', np.asarray(p.map(jpeg, Y)).flatten())
            TABLE = CHROMQUANTIZATIONTABLE
            with Pool(8) as p:
                compressed_cb = array.array('b', np.asarray(p.map(jpeg, CB)).flatten())
                compressed_cr = array.array('b', np.asarray(p.map(jpeg, CR)).flatten())

            q, qc = quant.flatten(), CHROMQUANTIZATIONTABLE.flatten()
            quantization_tables = array.array('b', q) + array.array('b', qc)
            dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
            compressed_data = quantization_tables + dim + compressed_y + compressed_cb + compressed_cr

            size, filename = EntropyReduction.bz2(compressed_data, resulting_file)
            compressed_file = resulting_file + '.bz2'

    print("--- %s seconds ---" % (time.time() - start_time))

