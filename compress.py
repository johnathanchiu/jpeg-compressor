from compressor.EntropyReduction import *
from JPEG.utils import *
from JPEG.binutils import *

from tqdm import tqdm

import imageio
import array

import time


def compress_image(image, file_name):

    def compress(image, debug=False, c_layer=False):
        image = image.copy().astype(float)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print(image); print()
        list_of_patches = split(matrix_multiple_of_eight(image) - 128, 8, 8)
        pbar = tqdm(list_of_patches)
        if debug:
            for x in list_of_patches:
                ext(capture(zig_zag(quantize(dct_2d(x, True), debug=True, c_layer=c_layer), debug=True),
                            c_layer=c_layer))
        else:
            for x in pbar:
                pbar.set_description("Running modified jpeg compression")
                ext(capture(zig_zag(quantize(dct_2d(x), c_layer=c_layer)), c_layer=c_layer))
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    o_length, o_width = image[:, :, 0].shape
    # print()
    # print("original file dimensions: ", o_length, o_width); print()

    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Converting image sample space RGB -> YCbCr")
        YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width],\
                (YCBCR[:, :, 1])[:o_length, :o_width],\
                (YCBCR[:, :, 2])[:o_length, :o_width]

    # Y, Cb, Cr = (YCBCR[:, :, 0])[:1000, :1000], \
    #             (YCBCR[:, :, 1])[:1000, :1000], \
    #             (YCBCR[:, :, 2])[:1000, :1000]

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)
    # print("padded image dimensions: ", p_length, p_width); print()
    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8),
                convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8),
               convertInt(dimensions[24:32], bits=8)]

    # padding = convertBin(p_length - c_length, bits=8) + \
    #           convertBin(p_length - c_width, bits=8)

    compressedY = compress(Y, debug=False)
    compressedCb = compress(Cb, debug=False, c_layer=True)
    compressedCr = compress(Cr, debug=False, c_layer=True)

    dim = array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = dim + compressedY + compressedCb + compressedCr
    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("writing file with entropy compressor")
        orig_size, size, filename = EntropyReduction.bz2(compressed, file_name)

    return orig_size, size, filename


if __name__ == '__main__':
    start_time = time.time()
    # print(start_time); print()
    root_path = "/Users/johnathanchiu/Documents/CompressionPics/tests/"  # enter file path of image
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="image name")
    # ap.add_argument("-c", "--compressed", required=True,
    #                 help="compressed file name")
    # args = vars(ap.parse_args())
    # image_name, compressed_file = args["image"], args["compressed"]
    # compressed_file_name = root_path + "compressed/fileSizes/" + compressed_file
    if root_path is None:
        image_name, compressed_file = input("Image path (You can set a root directory in the code): "), \
                                      input("Compressed file name (whatever you want to name the bz2 compressed file): ")
        compressed_file_name = compressed_file
        image = imageio.imread(image_name)
    else:
        image_name, compressed_file = input("Image path: "), \
                                      input("Compressed file name (whatever you want to name the bz2 compressed file): ")
        compressed_file_name = root_path + 'compressed/' + 'fileSizes/' + compressed_file
        image = imageio.imread(root_path + "tests/" + image_name)
    # file_size, size, filename, mo_filesize = compress_image(image, compressed_file_name)
    file_size, size, filename = compress_image(image, compressed_file_name)
    print()
    print("file size after (entropy) compression: ", size)
    # print("middle out reduced the file: ", mo_filesize - size * 8, "bits")
    print("file reduction percentage: ", (1 - ((file_size - size) / file_size)) * 100, "%")
    print("compression converges, new file name: ", filename)
    print("--- %s seconds ---" % (time.time() - start_time))

