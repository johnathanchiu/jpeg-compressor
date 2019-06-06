from PIL import Image

import numpy as np
from scipy.ndimage import *

from tqdm import tqdm
import time

from JPEG.utils import *
from JPEG.binutils import *
from compressor.EntropyReduction import *


def decompress_image(file_name):

    def decompress(input, dimx=0, dimy=0, debug=False, c_layer=False):
        input = np.asarray(list(input))
        if c_layer:
            compressed_split = [input[i:i + 8] for i in range(0, len(input), 8)]
        else:
            compressed_split = [input[i:i + 16] for i in range(0, len(input), 16)]
        image_partitions = []
        pbar = tqdm(compressed_split)
        append = image_partitions.append
        if debug: print(compressed_split); print()
        if debug:
            for x in compressed_split:
                idct_2d(undo_quantize(zig_zag_reverse(rebuild(x)), debug=True, c_layer=c_layer), debug=True)
        else:
            for x in pbar:
                pbar.set_description("Running modified jpeg decompression")
                append(idct_2d(undo_quantize(zig_zag_reverse(rebuild(x)), c_layer=c_layer)))
        if debug: print(image_partitions); print()
        pbar2 = tqdm(range(1))
        for _ in pbar2:
            pbar2.set_description("Merging blocks back to form whole image")
            image = merge_blocks(image_partitions, dimx, dimy)
        if debug: print(image); print()
        if debug: print("image: ", np.round(image + 128))
        return image + 128

    # def decompress_bitset(bitset):
    #     decompressed = MiddleOut.decompress(bitset, 1)
    #     return convertInt_list(decompressed)

    # compressed_bitset = readFile()
    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Reading bits from file using entropy decompressor")
        compressed_bitset = EntropyReduction.bz2_unc(file_name)

    # p_length, p_width = convertInt(compressed_bitset[:16], bits=16), \
    #                     convertInt(compressed_bitset[16:32], bits=16)

    p_length = convertInt(convertBin(compressed_bitset[0], bits=8) + convertBin(compressed_bitset[1], bits=8), bits=16)
    p_width = convertInt(convertBin(compressed_bitset[2], bits=8) + convertBin(compressed_bitset[3], bits=8), bits=16)

    s_length, s_width = int(p_length / 8), int(p_width / 8)
    # length, width = p_length - convertInt(compressed_bitset[32:40], bits=8), \
    #                 p_width - convertInt(compressed_bitset[40:48], bits=8)

    length, width = p_length - compressed_bitset[4], p_width - compressed_bitset[5]

    result_bytes = compressed_bitset[6:]
    # result_bytes = decompress_bitset(compressed_bitset[48:])

    no_of_values, no_of_values_cr = int((p_length * p_width) / 4), int((p_length * p_width) / 8)

    compressedY, compressedCb, compressedCr = result_bytes[:no_of_values], \
                                              result_bytes[no_of_values:no_of_values+no_of_values_cr], \
                                              result_bytes[no_of_values+no_of_values_cr:]

    newY, newCb, newCr = decompress(compressedY, dimx=s_length, dimy=s_width, debug=False), \
                         decompress(compressedCb, dimx=s_length, dimy=s_width, debug=False, c_layer=True), \
                         decompress(compressedCr, dimx=s_length, dimy=s_width, debug=False, c_layer=True)

    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Converting image sample space YCbCr -> RGB")
        rgbArray = np.flip(ycbcr2rgb(np.array([newY[0:length, 0:width], newCb[0:length, 0:width],
                                newCr[0:length, 0:width]]).T), axis=1)
        rgbArray = rotate(rgbArray, 90)

    img = Image.fromarray(rgbArray)
    img.save(image_save, "PNG", optimize=True)


if __name__ == '__main__':
    start_time = time.time()
    # print(start_time); print()
    root_path = None  # set root directory of project file
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-c", "--compressed", required=True,
    #                 help="compressed file name")
    # ap.add_argument("-d", "--decompressed", required=True,
    #                 help="decompressed image")
    # args = vars(ap.parse_args())
    # compressed_file, decompressed_image = args[0], args[1]
    if root_path is None:
        compressed_file, decompressed_image = input("Compressed file path without extension (You can set a root directory in the code): "), \
                                          input("Name of decompressed image without extension (You can set a root directory in the code): ")
        image_save = decompressed_image + ".png"
        compressed_file_name = compressed_file
    else:
        compressed_file, decompressed_image = input("Compressed file path without extension: "), \
                                              input("Name of decompressed image without extension: ")
        image_save = root_path + "compressed/testCases/" + decompressed_image + ".png"
        compressed_file_name = root_path + "compressed/fileSizes/" + compressed_file
    print();
    decompress_image(compressed_file_name)
    print(); print("Decompression converged, your file is at: ", image_save)
    print("--- %s seconds ---" % (time.time() - start_time))
