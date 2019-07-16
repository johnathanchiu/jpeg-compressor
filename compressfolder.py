from compressor.EntropyReduction import EntropyReduction
from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim
import imageio

from tqdm import tqdm
import argparse

import random
import array
import math
import time
import os



def compress_image(image, file_name):

    def compress(image, qual=64, count=1, debug=False, c=False):
        image_copy = image.copy().astype(np.int16)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print("original image: ", repr(image)); print()
        list_of_patches = split((matrix_multiple_of_eight(image_copy - 128)).astype(np.int8), 8, 8)
        if debug:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x, debug=True), debug=True, c=c), debug=True),
                             values=qual, sample_percentage=SAMPLE_RATIO, c=c)))
        else:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x), c=c)), values=qual, sample_percentage=SAMPLE_RATIO, c=c)))
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    def SSIM(photo, photo_x, photo_y, sample_area=200, c=False, debug=False, resample=False):
        if resample: print("Resampling with new area, previous patch was bad")
        assert photo_x >= 128 or photo_y >= 128, "Photo too small to run SSIM metric, compression diverges"
        grab_x, grab_y = int(photo_x // random.uniform(2, 5)), int(photo_y // random.uniform(2, 5))
        original_sample = np.array(photo[grab_x:grab_x + sample_area, grab_y:grab_y + sample_area], dtype=np.int16)
        ranging = range(10, 64)
        last_metric, rep = 0, 0
        for i in ranging:
            compressed_data, partitions = array.array('b', []), []
            list_of_patches = split((original_sample.copy() - 128).astype(np.int8), 8, 8)
            for x in list_of_patches:
                comp = capture(zig_zag(quantize(dct_2d(x), c=c)), values=i)
                compressed_data.extend(comp)
            compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
            for y in compressed_split:
                samples = idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)), c=c)) + 128
                partitions.append(samples)
            index = merge_blocks(partitions, int(sample_area/8), int(sample_area/8)).astype(np.uint8)
            metric = ssim(original_sample, index, data_range=index.max() - index.min())
            if debug: print((0 <= index).all() and (index <= 255).all()); print(metric)
            if math.isnan(metric):
                return SSIM(photo, photo_x, photo_y, sample_area=sample_area, c=c, resample=True)
            if metric > 0.97:
                return i, metric
            if abs(last_metric - metric) < 0.0000000001:
                if metric > 0.93:
                    return i, metric
                return SSIM(photo, photo_x, photo_y, sample_area=sample_area, c=c,  resample=True)
            rep += 1
            if rep == 4: last_metric = metric; rep = 0
        return 64, metric

    o_length, o_width = image[:, :, 0].shape

    YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width], \
                (YCBCR[:, :, 2])[:o_length, :o_width]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)

    values_to_keep, metric = SSIM(Cr, p_length, p_width, sample_area=SAMPLE_AREA, c=False, debug=debug)
    if values_to_keep % 2 != 0:
        values_to_keep += 1
    print("Number of samples (out of 64) to keep at metric " + str(metric) + ": ", values_to_keep)

    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, count=1)
    compressedCb = compress(Cb, qual=values_to_keep, count=2, c=True)
    compressedCr = compress(Cr, qual=values_to_keep, count=3, c=True)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = dim + compressedY + compressedCb + compressedCr
    size, filename = EntropyReduction.bz2(compressed, file_name)


if __name__ == '__main__':
    SAMPLE_AREA = 128; SAMPLE_RATIO = 1
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="image folder without trailing '/'")
    ap.add_argument("-c", "--compressed", required=True,
                    help="compressed file without trailing '/'")
    args = ap.parse_args()
    start_time = time.time()
    pbar = tqdm(os.listdir(args.images))
    for file in pbar:
        pbar.set_description("running compressor on all images in folder")
        filename = os.fsdecode(file)
        resulting_file = args.compressed + '/' + os.path.splitext(filename)[0]
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = imageio.imread(args.images+'/'+filename)
            compress_image(image, resulting_file)
    print("--- %s seconds ---" % (time.time() - start_time))

