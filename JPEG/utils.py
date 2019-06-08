from scipy import fftpack
from scipy import *
import numpy as np


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    return np.array([np.array(array[x:x + nrows, y:y + ncols], dtype=np.int16) for x in range(0, array.shape[0], nrows)
                     for y in range(0, array.shape[1], ncols)], dtype=np.int16)


# converts rgb to ycbcr colorspace
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


# converts the ycbcr colorspace back to rgb
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


# padding the image
def matrix_multiple_of_eight(image):
    while (len(image) % 8) > 0:
        pad = np.zeros(len(image[0])).reshape(1, len(image[0]))
        image = np.r_[image, pad]
    while (len(image[0]) % 8) > 0:
        pad = np.zeros(len(image)).reshape(len(image), 1)
        image = np.c_[image, pad]
    return image


def calc_matrix_eight_size(image_layer):
    row_count, col_count = image_layer.shape[0], image_layer.shape[1]
    while (row_count % 8) > 0:
        row_count += 1
    while (col_count % 8) > 0:
        col_count += 1
    return row_count, col_count


# grab top row of 8 by 8 and 0:4
def capture(image_patch, values=64, c_layer=False):
    image_patch = image_patch.flatten()
    if c_layer:
        return image_patch[:int(values * 8 / 10)].astype(int)
    return image_patch[:int(values)].astype(int)


def rebuild(image):
    return np.append(image, [0]*(64-len(image))).reshape((8, 8))


def zig_zag(input_matrix, block_size=8, debug=False):
    if debug: print(np.round(input_matrix)); print()
    z = np.empty([block_size * block_size])
    index = -1
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i - j]
            else:
                z[index] = input_matrix[i - j, j]
    z = z.reshape((8, 8), order='C')
    if debug: print("zig zag: ", np.round(z)); print()
    return np.round(z)


def zig_zag_reverse(input_matrix, block_size=8, debug=False):
    input_matrix = np.squeeze(input_matrix.reshape((1, 64), order='C'))
    output_matrix = np.empty([block_size, block_size])
    index = -1
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                output_matrix[j, i - j] = input_matrix[index]
            else:
                output_matrix[i - j, j] = input_matrix[index]
    if debug: print("zig zag reverse: ", output_matrix); print()
    return output_matrix


def dct_2d(image, debug=False):
    if debug: print(image); print()
    image.astype(float)
    if debug: print("dct: ", np.round(fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho'))); print()
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def idct_2d(image, debug=False):
    if debug: print(image); print()
    image.astype(float)
    if debug: print("idct: ", fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')); print()
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def merge_blocks(input_list, rows, columns):
    all_rows_concatenated = []
    append = all_rows_concatenated.append
    for row in range(rows):
        this_row_items = input_list[(columns * row):(columns * (row + 1))]
        append(np.concatenate(this_row_items, axis=1))
    output_matrix = np.concatenate(all_rows_concatenated, axis=0)
    return output_matrix


def quantize(input, debug=False, c_layer=False):
    if debug: print(input); print()
    q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
    q_c = np.array([[8, 4, 3, 5, 9, 12, 31, 37],
                    [3, 2, 8, 3, 16, 35, 36, 33],
                    [4, 4, 1, 2, 24, 34, 41, 34],
                    [2, 1, 5, 17, 31, 52, 48, 37],
                    [6, 13, 22, 34, 41, 65, 62, 46],
                    [14, 21, 33, 38, 49, 62, 68, 55],
                    [29, 38, 47, 52, 62, 73, 72, 61],
                    [43, 55, 57, 59, 67, 60, 62, 59]])
    if debug: print("quantize: ", input/q); print()
    if c_layer:
        return (input / q_c).astype(np.int8)
    return (input / q).astype(np.int8)


def undo_quantize(input, debug=False, c_layer=False):
    if debug: print(input); print()
    q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])
    q_c = np.array([[8, 4, 3, 5, 9, 12, 31, 37],
                    [3, 2, 8, 3, 16, 35, 36, 33],
                    [4, 4, 1, 2, 24, 34, 41, 34],
                    [2, 1, 5, 17, 31, 52, 48, 37],
                    [6, 13, 22, 34, 41, 65, 62, 46],
                    [14, 21, 33, 38, 49, 62, 68, 55],
                    [29, 38, 47, 52, 62, 73, 72, 61],
                    [43, 55, 57, 59, 67, 60, 62, 59]])
    if debug: print("undo quantize: ", input*q); print()
    if c_layer:
        return input.astype(np.int16) * q_c
    return input.astype(np.int16) * q


