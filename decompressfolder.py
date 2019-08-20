import imageio
from scipy.ndimage import *
from multiprocessing import Pool

from tqdm import tqdm
import argparse
import time
import os

from jpeg.utils import *
from jpeg.binutils import convertInt, convertBin
from compressor.EntropyReduction import EntropyReduction


SAMPLE_RATIO, TABLE = 1.0, QUANTIZATIONTABLE


def jpeg_decompress(partition):
    return (idct_2d(undo_quantize(zig_zag_reverse(rebuild(partition)), table=TABLE)) + 128).astype(np.uint8)


def image_attributes(compressed):
    table = np.asarray(compressed[:64], dtype=np.float16).reshape(8, 8)
    tablecr = np.asarray(compressed[64:128], dtype=np.float16).reshape(8, 8)
    quality_metric = compressed[128]
    p_length = convertInt(convertBin(compressed[129],bits=8)+convertBin(compressed[130],bits=8),bits=16)
    p_width = convertInt(convertBin(compressed[131],bits=8)+convertBin(compressed[132],bits=8),bits=16)
    length, width = p_length - compressed[133], p_width - compressed[134]
    val = int(p_length*p_width / 64*int(quality_metric*SAMPLE_RATIO))
    val_cr = int(p_length*p_width / 64*int(quality_metric*SAMPLE_RATIO))
    return table, tablecr, p_length, p_width, length, width, val, val_cr


if __name__ == '__main__':
    SAMPLE_RATIO = 1
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--compressed", required=True,
                    help="folder of compressed")
    ap.add_argument("-d", "--directory", required=True,
                    help="decompressed folder")
    args = ap.parse_args()
    start_time = time.time()
    pbar = tqdm(os.listdir(args.compressed), desc="running decompressor on all images in folder")
    for file in pbar:
        filename = os.fsdecode(file)
        if args.iden == 'Y' or args.iden == 'y': resulting_file = args.directory+os.path.splitext(filename)[0] + '.jpg'
        else: image_save = resulting_file = args.directory+os.path.splitext(filename)[0] + '.png'

        if filename.endswith(".bz2"):
            comp = EntropyReduction.bz2_unc(args.compressed+filename)

            data = comp[135:]
            tab, tabcr, p_length, p_width, length, width, val, val_cr = image_attributes(comp)
            compressedY, compressedCb, compressedCr = data[:val], data[val:val+val_cr], data[val+val_cr:val+(2*val_cr)]

            TABLE = tab
            with Pool(8) as p:
                split_y = [np.array(compressedY[i:i+int(comp[128]*SAMPLE_RATIO)], dtype=np.int8)
                           for i in range(0, len(compressedY), int(comp[128]*SAMPLE_RATIO))]
                y = merge_blocks(p.map(jpeg_decompress, split_y), int(p_length / 8), int(p_width / 8))
            TABLE = tabcr
            with Pool(8) as p:
                split_cb = [np.array(compressedCb[i:i+int(comp[128]*SAMPLE_RATIO)], dtype=np.int8)
                            for i in range(0, len(compressedCb), int(comp[128]*SAMPLE_RATIO))]
                split_cr = [np.array(compressedCr[i:i+int(comp[128]*SAMPLE_RATIO)], dtype=np.int8)
                            for i in range(0, len(compressedCr), int(comp[128]*SAMPLE_RATIO))]
                cb = merge_blocks(p.map(jpeg_decompress, split_cb), int(p_length / 8), int(p_width / 8))
                cr = merge_blocks(p.map(jpeg_decompress, split_cr), int(p_length / 8), int(p_width / 8))

            YCBCR = np.array([y[0:length, 0:width], cb[0:length, 0:width], cr[0:length, 0:width]]).T
            rgbArray = ycbcr2rgb(np.flip(YCBCR, axis=1)); rgbArray = rotate(rgbArray, 90)

            imageio.imwrite(resulting_file, rgbArray, quality=100)

    print("--- %s seconds ---" % (time.time() - start_time))
