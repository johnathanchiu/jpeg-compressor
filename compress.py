from compressor.EntropyReduction import EntropyReduction
from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim

from tqdm import tqdm
import argparse
import random

import imageio
import array
import os

import time


def compress_image(image, file_name, debug=False):

    def compress(image, qual=64, count=1, debug=False, c=False):
        image_copy = image.copy().astype(float)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print(image); print()
        list_of_patches = split((matrix_multiple_of_eight(image_copy) - 128).astype(np.int8), 8, 8)
        pbar = tqdm(list_of_patches)
        if debug:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x, debug=True), debug=True, c=c), debug=True),
                             values=qual, sample_percentage=SAMPLE_RATIO, c=c)))
        else:
            for x in pbar:
                descrip = "Running modified jpeg compression " + str(count) + " / 3"
                pbar.set_description(descrip)
                ext((capture(zig_zag(quantize(dct_2d(x), c=c)), values=qual, sample_percentage=SAMPLE_RATIO,
                             c=c)))
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    def SSIM(photo, photo_x, photo_y, sample_area=200, c=False):
        assert photo_x >= 512 or photo_y >= 512, "Photo too small to run SSIM metric, compression diverges"
        grab_x, grab_y = int(photo_x / random.uniform(2, 8)), int(photo_y / random.uniform(2, 8))
        original_sample = np.array(photo[grab_x:grab_x + sample_area, grab_y:grab_y + sample_area], dtype=np.int16)
        pbar = tqdm(range(16, 64))
        last_metric = 0
        for i in pbar:
            compressed_data, partitions = array.array('b', []), []
            pbar.set_description("Running SSIM metric quality, 16 through 64 sampled weights")
            list_of_patches = split(original_sample.copy() - 128, 8, 8)
            for x in list_of_patches:
                comp = capture(zig_zag(quantize(dct_2d(x), c=c)), values=i)
                compressed_data.extend(comp)
            compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
            for y in compressed_split:
                samples = idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)), c=c)) + 128
                partitions.append(samples)
            index = merge_blocks(partitions, int(sample_area/8), int(sample_area/8))
            metric = ssim(original_sample.flatten(), index.flatten(), data_range=index.max() - index.min())
            if metric > 0.99 or (abs(last_metric - metric) < 0.0000000001):
                return i
            last_metric = metric
        return 64

    o_length, o_width = image[:, :, 0].shape

    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Converting image sample space RGB -> YCbCr")
        YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width], (YCBCR[:, :, 2])[:o_length, :o_width]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)

    values_to_keep = SSIM(Cr, p_length, p_width, sample_area=SAMPLE_AREA, c=False)
    if values_to_keep % 2 != 0:
        values_to_keep += 1
    print("Number of samples (out of 64) to keep: ", values_to_keep)

    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, count=1, debug=debug)
    compressedCb = compress(Cb, qual=values_to_keep, count=2, debug=debug, c=True)
    compressedCr = compress(Cr, qual=values_to_keep, count=3, debug=debug, c=True)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = dim + compressedY + compressedCb + compressedCr
    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("writing file with entropy compressor")
        size, filename = EntropyReduction.bz2(compressed, file_name)

    return size, filename


if __name__ == '__main__':
    SAMPLE_RATIO = 1; SAMPLE_AREA = 200
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--image", required=True,
                    help="Image name with path")
    ap.add_argument('-c', "--compressed", default='./',
                    help="Folder to save compressed file")
    args = ap.parse_args()
    image_path, compressed = args.image, args.compressed
    start_time = time.time()
    image = imageio.imread(image_path)
    _, tail = os.path.split(image_path)
    size, filename = compress_image(image, compressed+os.path.splitext(tail)[0])
    file_size = os.stat(image_path).st_size
    print()
    print("file size after (entropy) compression: ", size)
    print("file reduction percentage (new file size / old file size): ", (size / file_size) * 100, "%")
    print("compression converges, new file name: ", filename)
    print("--- %s seconds ---" % (time.time() - start_time))

