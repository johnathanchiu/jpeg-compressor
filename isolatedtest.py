from skimage.measure._structural_similarity import compare_ssim as ssim

from JPEG.utils import *
import numpy as np

DEBUG_PRINT = True

patch = np.array([[206, 136,  59,  37,  40,  29,  43,  43], [203, 178, 100,  29,  31,  29,  40,  45],
                  [199, 189, 138,  63,  25,  27,  41,  46], [193, 195, 173, 104,  33,  35,  40,  40],
                  [194, 197, 197, 137,  62,  24,  37,  39], [186, 206, 194, 180, 103,  19,  23,  45],
                  [189, 193, 198, 201, 133,  64,  26,  23],[189, 181, 199, 202, 166, 100,  24,  28]], dtype=np.int16)


for i in range(1):
    # patch = (np.random.randint(0, 20, size=(8, 8)) - 128).astype(np.int8)
    patch = (patch - 128).astype(np.int8)
    if DEBUG_PRINT: print("original:", patch); print()
    dct = dct_2d(patch)
    if DEBUG_PRINT: print("dct:", dct); print()
    quant = quantize(dct, c=False)
    if DEBUG_PRINT: print("quantized:", quant); print()
    zigzag = zig_zag(quant, block_size=8)
    if DEBUG_PRINT: print("zig zag:", zigzag); print()
    cap = zigzag[:10]
    zigzag = np.append(cap, [0]*(64-len(cap)))
    zigzagr = zig_zag_reverse(zigzag, block_size=8)
    if DEBUG_PRINT: print("zig zag reverse:", zigzagr); print()
    unquant = undo_quantize(zigzagr, c=False)
    if DEBUG_PRINT: print("undo quantization:", unquant); print()
    idct = idct_2d(unquant)
    if DEBUG_PRINT: print("idct:", idct); print()
    print("resulting: ", (idct + 128).astype(np.uint8)); print()
    print("original: ", patch + 128); print()

    orig = (patch.astype(np.int16) + 128).astype(np.uint8)
    final = (idct + 128).astype(np.uint8)
    print(orig.dtype, final.dtype)
    qual = ssim(orig, final, index=final.max() - final.min())
    print("SSIM: ", qual); print()
    if qual < 0.90:
        print("bad quality at test num: ", i)
        print("testing cause of faults...")
        print("SSIM between undo quantized and dct: ", ssim(dct, unquant, index=unquant.max() - unquant.min()))
        print("dct: ", np.round(dct).astype(np.float16)); print(); print("undo quantize: ", unquant)
        break


