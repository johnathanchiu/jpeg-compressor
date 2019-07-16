from compressor.EntropyReduction import EntropyReduction
from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim

from tqdm import tqdm
import argparse
import random
import math

import imageio
import array
import os

import time


def compress_image(image, file_name, debug=False):

    def compress(image, qual=64, sampratio=1.0, table=QUANTIZATIONTABLE, count=1, debug=False):
        image_copy = image.copy().astype(np.int16)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print("original image: ", repr(image)); print()
        list_of_patches = split((matrix_multiple_of_eight(image_copy - 128)).astype(np.int8), 8, 8)
        if debug:
            for x in list_of_patches:
                ext((capture(zig_zag(quantize(dct_2d(x, debug=True), table=table, debug=True), debug=True),
                             values=qual, sample_percentage=sampratio)))
        else:
            pbar = tqdm(list_of_patches)
            descrip = "Running modified jpeg compression " + str(count) + " / 3"
            pbar.set_description(descrip)
            [ext((capture(zig_zag(quantize(dct_2d(x), table=table)), values=qual, sample_percentage=sampratio))) for x in pbar]
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    def SSIM(photo, photo_x, photo_y, area=200, table=QUANTIZATIONTABLE, resample=False, debug=False):
        if resample: print(); print("Resampling with new area, previous patch was bad")
        assert photo_x >= 64 or photo_y >= 64, "Photo too small to run SSIM metric, compression diverges"
        assert area % 8 == 0, "Invalid sampling area make sure sample area is equally divisible by 8"
        grab_x, grab_y = int(photo_x // random.uniform(1.5, 4)), int(photo_y // random.uniform(1.5, 4))
        original_sample = np.array(photo[grab_x:grab_x + area, grab_y:grab_y + area], dtype=np.int16)
        pbar = tqdm(range(8, 64))
        if debug: pbar = range(8, 64)
        last_metric, rep = 0, 0
        for i in pbar:
            compressed_data, partitions = array.array('b', []), []
            ext = compressed_data.extend; app = partitions.append
            if not debug: pbar.set_description("Running SSIM metric quality, 8 through 64 sampled weights")
            list_of_patches = split((original_sample.copy() - 128).astype(np.int8), 8, 8)
            [ext(capture(zig_zag(quantize(dct_2d(x), table=table)), values=i)) for x in list_of_patches]
            compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
            [app(idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)), table=table)) + 128) for y in compressed_split]
            index = merge_blocks(partitions, int(area/8), int(area/8)).astype(np.uint8)
            metric = ssim(original_sample, index, data_range=index.max() - index.min())
            if debug: print(metric)
            if math.isnan(metric): return SSIM(photo, photo_x, photo_y, area=area, resample=True, debug=debug)
            if metric < 0.7:
                return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.1), resample=True, debug=debug)
            if metric > 0.975: return i, metric, table
            if abs(last_metric - metric) < 0.0000000001:
                if metric > 0.93:
                    return i - rep, metric, table
                return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.1), resample=True, debug=debug)
            rep += 1
            if rep == 5: last_metric = metric; rep = 0
        if metric < 0.93:
            if debug: print('recompiling quantization table')
            return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.2), resample=True, debug=debug)
        return 64, metric, table

    o_length, o_width = image[:, :, 0].shape

    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Converting image sample space RGB -> YCbCr")
        YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width],\
                (YCBCR[:, :, 2])[:o_length, :o_width]

    ''' line(s) below are for debugging '''
    if debug:
        debug_grab = 8
        randl, randw = 1884, 1320
        Y, Cb, Cr = (YCBCR[:, :, 0])[randl:randl + debug_grab, randw:randw + debug_grab], \
                    (YCBCR[:, :, 1])[randl:randl + debug_grab, randw:randw + debug_grab], \
                    (YCBCR[:, :, 2])[randl:randl + debug_grab, randw:randw + debug_grab]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)

    values_to_keep, metric, quant = SSIM(Cr, p_length, p_width, area=SAMPLE_AREA, table=QUANTIZATIONTABLE, debug=DEBUG)
    if quant[0][0] < 8: quant[0][0] = 8
    if debug: print("quantization table:"); print(quant)
    ''' line below is for debugging '''
    if debug: values_to_keep, metric = 64, 1
    if values_to_keep % 2 != 0:
        values_to_keep += 1
    print("Number of samples (out of 64) to keep at metric " + str(metric) + ": ", values_to_keep)

    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, table=quant, count=1, debug=debug)
    compressedCb = compress(Cb, qual=values_to_keep, sampratio=SAMPLE_RATIO,
                            table=CHROMQUANTIZATIONTABLE, count=2, debug=debug)
    compressedCr = compress(Cr, qual=values_to_keep, sampratio=SAMPLE_RATIO,
                            table=CHROMQUANTIZATIONTABLE, count=3, debug=debug)

    q, qc = quant.flatten(), CHROMQUANTIZATIONTABLE.flatten()
    quantization_tables = array.array('b', q) + array.array('b', qc)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = quantization_tables + dim + compressedY + compressedCb + compressedCr
    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("writing file with entropy compressor")
        size, filename = EntropyReduction.bz2(compressed, file_name)

    return size, filename


if __name__ == '__main__':
    DEBUG = False
    SAMPLE_RATIO = 1; SAMPLE_AREA = 152
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--image", required=True, help="Image name with path")
    ap.add_argument('-c', "--compressed", default='./', help="Folder to save compressed file")
    args = ap.parse_args()
    image_path, compressed = args.image, args.compressed
    start_time = time.time()
    image = imageio.imread(image_path)
    _, tail = os.path.split(image_path)
    size, filename = compress_image(image, compressed+os.path.splitext(tail)[0], debug=DEBUG)
    file_size = os.stat(image_path).st_size
    print()
    print("file size after (entropy) compression: ", size)
    print("file reduction percentage (new file size / old file size): ", (size / file_size) * 100, "%")
    print("compression converges, new file name: ", filename)
    print("--- %s seconds ---" % (time.time() - start_time))

