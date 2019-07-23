import imageio
from scipy.ndimage import *

from tqdm import tqdm
import argparse
import time
import os

from jpeg.utils import *
from jpeg.binutils import convertInt, convertBin
from compressor.EntropyReduction import EntropyReduction


def decompress_image(file_name, image_save):

    def decompress(input, dimx=0, dimy=0, qual=64, table=QUANTIZATIONTABLE):
        compressed_split = np.array([np.array(input[i:i+qual], dtype=np.int8)
                                     for i in range(0, len(input), qual)], dtype=np.int8)
        image_partitions = []; append = image_partitions.append
        for x in compressed_split:
            append((idct_2d(undo_quantize(zig_zag_reverse(rebuild(x)), table=table)) + 128).astype(np.uint8))
        image = merge_blocks(image_partitions, dimx, dimy)
        return image

    compressed_bitset = EntropyReduction.bz2_unc(file_name)
    table = np.asarray(compressed_bitset[:64], dtype=np.float16).reshape(8, 8)
    tablecr = np.asarray(compressed_bitset[64:128], dtype=np.float16).reshape(8, 8)
    quality_metric = compressed_bitset[128]

    p_length = convertInt(convertBin(compressed_bitset[129], bits=8) +
                          convertBin(compressed_bitset[130], bits=8), bits=16)
    p_width = convertInt(convertBin(compressed_bitset[131], bits=8) +
                         convertBin(compressed_bitset[132], bits=8), bits=16)
    s_length, s_width = int(p_length / 8), int(p_width / 8)
    length, width = p_length - compressed_bitset[133], p_width - compressed_bitset[134]

    result_bytes = compressed_bitset[135:]
    no_of_values, no_of_values_cr = int(p_length * p_width / 64 * quality_metric), \
                                    int(p_length * p_width / 64 * int(quality_metric * SAMPLE_RATIO))

    compressedY = result_bytes[:no_of_values]
    compressedCb = result_bytes[no_of_values:no_of_values+no_of_values_cr]
    compressedCr = result_bytes[no_of_values+no_of_values_cr:no_of_values+(2*no_of_values_cr)]

    newY = decompress(compressedY, dimx=s_length, dimy=s_width, qual=quality_metric, table=table)
    newCb = decompress(compressedCb, dimx=s_length, dimy=s_width, qual=int(quality_metric*SAMPLE_RATIO), table=tablecr)
    newCr = decompress(compressedCr, dimx=s_length, dimy=s_width, qual=int(quality_metric*SAMPLE_RATIO), table=tablecr)

    YCBCR = np.array([newY[0:length, 0:width], newCb[0:length, 0:width], newCr[0:length, 0:width]]).T
    rgbArray = ycbcr2rgb(np.flip(YCBCR, axis=1)); rgbArray = rotate(rgbArray, 90)

    imageio.imwrite(image_save, rgbArray, quality=100)


if __name__ == '__main__':
    SAMPLE_RATIO = 1
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--compressed", required=True,
                    help="folder of compressed")
    ap.add_argument("-d", "--directory", required=True,
                    help="decompressed folder")
    args = ap.parse_args()
    start_time = time.time()
    pbar = tqdm(os.listdir(args.compressed))
    for file in pbar:
        pbar.set_description("running decompressor on all images in folder")
        filename = os.fsdecode(file)
        resulting_file = args.directory+os.path.splitext(filename)[0] + '.jpg'
        if filename.endswith(".bz2"):
            decompress_image(args.compressed+filename, resulting_file)
    print("--- %s seconds ---" % (time.time() - start_time))
