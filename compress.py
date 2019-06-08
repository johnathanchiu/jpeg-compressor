from compressor.EntropyReduction import EntropyReduction
from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin

from skimage.measure._structural_similarity import compare_ssim as ssim

from tqdm import tqdm

import imageio
import array
import os

import time


def compress_image(image, file_name):

    def compress(image, qual=64, count=1, debug=False, c_layer=False):
        image_copy = image.copy().astype(np.uint8)
        compressed_data = array.array('b', [])
        ext = compressed_data.extend
        if debug: print(image); print()
        list_of_patches = split(matrix_multiple_of_eight(image_copy) - 128, 8, 8).astype(np.int8)
        pbar = tqdm(list_of_patches)
        if debug:
            for x in list_of_patches:
                ext(capture(zig_zag(quantize(dct_2d(x, True), debug=True, c_layer=c_layer), debug=True), values=qual,
                            c_layer=c_layer))
        else:
            for x in pbar:
                descrip = "Running modified jpeg compression " + str(count) + " / 3"
                pbar.set_description(descrip)
                ext(capture(zig_zag(quantize(dct_2d(x), c_layer=c_layer)), values=qual, c_layer=c_layer))
        if debug: print("compressed data: ", compressed_data); print()
        return compressed_data

    def SSIM(photo, photo_x, photo_y):
        assert photo_x >= 512 or photo_y >= 512, "Photo too small to run SSIM metric, compression diverges"
        grab_x, grab_y = int(photo_x / 2), int(photo_y / 2)
        original_sample = np.array(photo[grab_x:grab_x + 176, grab_y:grab_y + 176], dtype=np.int16)
        pbar = tqdm(range(18, 64))
        previous_metric = 0
        for i in pbar:
            compressed_data = array.array('b', [])
            partitions = []
            pbar.set_description("Running SSIM metric quality, 18 through 64 sampled weights")
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
                if metric > 0.98 or abs(previous_metric - metric) < 0.000001:
                    return i
                previous_metric = metric
        return 64

    o_length, o_width = image[:, :, 0].shape

    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("Converting image sample space RGB -> YCbCr")
        YCBCR = rgb2ycbcr(image)

    Y, Cb, Cr = (YCBCR[:, :, 0])[:o_length, :o_width], (YCBCR[:, :, 1])[:o_length, :o_width], \
                (YCBCR[:, :, 2])[:o_length, :o_width]

    values_to_keep = SSIM(Y, o_length, o_width)
    if values_to_keep % 2 != 0:
        values_to_keep += 1
    if values_to_keep <= 62:
        values_to_keep += 2

    print("Number of samples (out of 64) to keep: ", values_to_keep)

    c_length, c_width = Y.shape
    p_length, p_width = calc_matrix_eight_size(Y)
    # print("padded image dimensions: ", p_length, p_width); print()
    dimensions = convertBin(p_length, bits=16) + convertBin(p_width, bits=16)
    padding = [p_length - c_length, p_width - c_width]
    p_length = [convertInt(dimensions[:8], bits=8), convertInt(dimensions[8:16], bits=8)]
    p_width = [convertInt(dimensions[16:24], bits=8), convertInt(dimensions[24:32], bits=8)]
    keep = [values_to_keep]

    compressedY = compress(Y, qual=values_to_keep, count=1, debug=False)
    compressedCb = compress(Cb, qual=values_to_keep, count=2, debug=False, c_layer=True)
    compressedCr = compress(Cr, qual=values_to_keep, count=3, debug=False, c_layer=True)

    dim = array.array('b', keep) + array.array('b', p_length) + array.array('b', p_width) + array.array('b', padding)
    compressed = dim + compressedY + compressedCb + compressedCr
    pbar = tqdm(range(1))
    for _ in pbar:
        pbar.set_description("writing file with entropy compressor")
        size, filename = EntropyReduction.bz2(compressed, file_name)

    return size, filename


if __name__ == '__main__':
    start_time = time.time()
    # print(start_time); print()
    root_path = None  # enter file path of image
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
        compressed_file_name = root_path + compressed_file
        image = imageio.imread(root_path + image_name)
    size, filename = compress_image(image, compressed_file_name)
    file_size = os.stat(image_name).st_size
    print()
    print("file size after (entropy) compression: ", size)
    print("file reduction percentage (new file size / old file size): ", (size / file_size) * 100, "%")
    print("compression converges, new file name: ", filename)
    print("--- %s seconds ---" % (time.time() - start_time))

