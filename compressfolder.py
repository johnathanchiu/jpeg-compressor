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

    def compress(image, qual=64, sampratio=1.0, table=QUANTIZATIONTABLE):
        image_copy = image.copy().astype(np.int16)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        list_of_patches = split((matrix_multiple_of_eight(image_copy - 128)).astype(np.int8), 8, 8)
        for x in list_of_patches:
            ext((capture(zig_zag(quantize(dct_2d(x), table=table)), values=qual, sample_percentage=sampratio)))
        return compressed_data

    def SSIM(photo, photo_x, photo_y, area=200, table=QUANTIZATIONTABLE):
        assert photo_x >= 64 or photo_y >= 64, "Photo too small to run SSIM metric, compression diverges"
        assert area % 8 == 0, "Invalid sampling area make sure sample area is equally divisible by 8"
        grab_x, grab_y = int(photo_x // random.uniform(1.5, 4)), int(photo_y // random.uniform(1.5, 4))
        original_sample = np.array(photo[grab_x:grab_x + area, grab_y:grab_y + area], dtype=np.int16)
        pbar = range(8, 64)
        last_metric, rep = 0, 0
        for i in pbar:
            compressed_data, partitions = array.array('b', []), []
            list_of_patches = split((original_sample.copy() - 128).astype(np.int8), 8, 8)
            for x in list_of_patches:
                comp = capture(zig_zag(quantize(dct_2d(x), table=table)), values=i)
                compressed_data.extend(comp)
            compressed_split = [compressed_data[z:z + i] for z in range(0, len(compressed_data), i)]
            for y in compressed_split:
                samples = (idct_2d(undo_quantize(zig_zag_reverse(rebuild(y)), table=table)) + 128).astype(np.uint8)
                partitions.append(samples)
            index = merge_blocks(partitions, int(area/8), int(area/8)).astype(np.uint8)
            metric = ssim(original_sample, index, data_range=index.max() - index.min())
            if math.isnan(metric): return SSIM(photo, photo_x, photo_y, area=area)
            if metric < 0.7:
                return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.1))
            if metric > 0.975: return i, metric, table
            if abs(last_metric - metric) < 0.0000000001:
                if metric > 0.93:
                    return i - rep, metric, table
                return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.1))
            rep += 1
            if rep == 5: last_metric = metric; rep = 0
        if metric < 0.93:
            return SSIM(photo, photo_x, photo_y, area=area, table=np.round(table/1.2))
        return 64, metric, table

    o_length, o_width = image[:, :, 0].shape
    YCBCR = rgb2ycbcr(image)
    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width], \
                (YCBCR[:, :, 2])[:o_length, :o_width]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)

    values_to_keep, metric, quant = SSIM(Cr, p_length, p_width, area=SAMPLE_AREA, table=QUANTIZATIONTABLE)
    if quant[0][0] < 8: quant[0][0] = 8
    if values_to_keep % 2 != 0:
        values_to_keep += 1

    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, table=quant)
    compressedCb = compress(Cb, qual=values_to_keep, sampratio=SAMPLE_RATIO, table=CHROMQUANTIZATIONTABLE)
    compressedCr = compress(Cr, qual=values_to_keep, sampratio=SAMPLE_RATIO, table=CHROMQUANTIZATIONTABLE)

    q, qc = quant.flatten(), CHROMQUANTIZATIONTABLE.flatten()
    quantization_tables = array.array('b', q) + array.array('b', qc)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = quantization_tables + dim + compressedY + compressedCb + compressedCr
    EntropyReduction.bz2(compressed, file_name)


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

