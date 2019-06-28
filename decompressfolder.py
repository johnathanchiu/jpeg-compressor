from PIL import Image

from scipy.ndimage import *

from tqdm import tqdm
import time
import argparse

from JPEG.utils import *
from JPEG.binutils import convertInt, convertBin
from compressor.EntropyReduction import EntropyReduction


def decompress_image(file_name, image_save):

    def decompress(input, dimx=0, dimy=0, qual=64, c_layer=False, count=1, debug=False):
        if c_layer:
            compressed_split = np.array([np.array(input[i:i+int(qual * 8 / 10)], dtype=np.int8)
                                         for i in range(0, len(input), int(qual * 8 / 10))], dtype=np.int8)
        else:
            compressed_split = np.array([np.array(input[i:i+qual], dtype=np.int8)
                                         for i in range(0, len(input), qual)], dtype=np.int8)
        image_partitions = []
        append = image_partitions.append
        if debug: print(compressed_split); print()
        if debug:
            for x in compressed_split:
                append(idct_2d(undo_quantize(zig_zag_reverse(rebuild(x)), debug=True, c_layer=c_layer), debug=True))
        else:
            for x in compressed_split:
                append(idct_2d(undo_quantize(zig_zag_reverse(rebuild(x)), c_layer=c_layer)))
        if debug: print(image_partitions); print()
        image = merge_blocks(image_partitions, dimx, dimy)
        if debug: print(image); print()
        if debug: print("image: ", np.round(image + 128))
        return image + 128

    compressed_bitset = EntropyReduction.bz2_unc(file_name)

    quality_metric = compressed_bitset[0]
    p_length = convertInt(convertBin(compressed_bitset[1], bits=8) + convertBin(compressed_bitset[2], bits=8), bits=16)
    p_width = convertInt(convertBin(compressed_bitset[3], bits=8) + convertBin(compressed_bitset[4], bits=8), bits=16)
    s_length, s_width = int(p_length / 8), int(p_width / 8)
    length, width = p_length - compressed_bitset[5], p_width - compressed_bitset[6]

    result_bytes = compressed_bitset[7:]
    no_of_values, no_of_values_cr = int((p_length * p_width) / 64 * quality_metric), \
                                    int((p_length * p_width) / 64 * (int(quality_metric * 8 / 10)))

    compressedY = result_bytes[:no_of_values]
    compressedCb = result_bytes[no_of_values:no_of_values+no_of_values_cr]
    compressedCr = result_bytes[no_of_values+no_of_values_cr:no_of_values+(2*no_of_values_cr)]

    newY = decompress(compressedY, dimx=s_length, dimy=s_width, qual=quality_metric, c_layer=False, count=1)
    newCb = decompress(compressedCb, dimx=s_length, dimy=s_width, qual=quality_metric, c_layer=True, count=2)
    newCr = decompress(compressedCr, dimx=s_length, dimy=s_width, qual=quality_metric, c_layer=True, count=3)

    YCBCR = np.array([newY[0:length, 0:width], newCb[0:length, 0:width], newCr[0:length, 0:width]]).T
    rgbArray = ycbcr2rgb(np.flip(YCBCR, axis=1))
    rgbArray = rotate(rgbArray, 90)

    img = Image.fromarray(rgbArray)
    img.save(image_save, "PNG")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--compressed", required=True,
                    help="folder of compressed images without trailing '/'")
    ap.add_argument("-d", "--directory", required=True,
                    help="decompressed folder without trailing '/'")
    args = ap.parse_args()
    start_time = time.time()
    pbar = tqdm(os.listdir(args.directory))
    for file in pbar:
        pbar.set_description("running decompressor on all images in folder")
        filename = os.fsdecode(file)
        resulting_file = args.directory+'/'+os.path.splitext(filename)[0] + '.bmp'
        if filename.endswith(".bz2"):
            decompress_image(args.compressed+'/'+filename, resulting_file)
    print();
    print("--- %s seconds ---" % (time.time() - start_time))