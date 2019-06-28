from compressor.EntropyReduction import EntropyReduction
from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim

from tqdm import tqdm
import argparse

import imageio
import array
import os

import time


def compress_image(image, file_name):

    def compress(image, qual=64, debug=False, c_layer=False):
        image_copy = image.copy().astype(float)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print(image); print()
        list_of_patches = split((matrix_multiple_of_eight(image_copy) - 128).astype(np.int8), 8, 8)
        if debug:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x, debug=True), debug=True, c_layer=c_layer), debug=True),
                             values=qual, c_layer=c_layer)))
        else:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x), c_layer=c_layer)), values=qual, c_layer=c_layer)))
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    def SSIM(photo, photo_x, photo_y):
        assert photo_x >= 512 or photo_y >= 512, "Photo too small to run SSIM metric, compression diverges"
        grab_x, grab_y = int(photo_x / random.uniform(2, 4)), int(photo_y / random.uniform(2, 4))
        original_sample = np.array(photo[grab_x:grab_x + 176, grab_y:grab_y + 176], dtype=np.int16)
        previous_metric = 0
        for i in range(10, 64):
            compressed_data = array.array('b', [])
            partitions = []
            list_of_patches = split(original_sample - 128, 8, 8)
            for x in list_of_patches:
                comp = capture(zig_zag(quantize(dct_2d(x))), values=i)
                compressed_data.extend(comp)
            compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
            for y in compressed_split:
                samples = idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)))) + 128
                partitions.append(samples)
            index = merge_blocks(partitions, int(176/8), int(176/8))
            metric = ssim(original_sample.flatten(), index.flatten(), data_range=index.max() - index.min())
            if i == 1:
                previous_metric = metric
            else:
                if metric > 0.98 or abs(previous_metric - metric) < 0.00001:
                    return i - 1
                previous_metric = metric
        return 64

    o_length, o_width = image[:, :, 0].shape

    YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width], \
                (YCBCR[:, :, 2])[:o_length, :o_width]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)

    values_to_keep = SSIM(Y, p_length, p_width)
    if values_to_keep % 2 != 0:
        values_to_keep += 1

    # print("padded image dimensions: ", p_length, p_width); print()
    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, debug=False)
    compressedCb = compress(Cb, qual=values_to_keep, debug=False, c_layer=True)
    compressedCr = compress(Cr, qual=values_to_keep, debug=False, c_layer=True)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = dim + compressedY + compressedCb + compressedCr
    EntropyReduction.bz2(compressed, file_name)


if __name__ == '__main__':
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

