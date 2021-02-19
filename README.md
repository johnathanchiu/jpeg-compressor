# JPEG-COMPRESSOR

This algorithm is a theoretical modification to the preexisting JPEG algorithm for reduced file sizes. The lossless
entropy algorithm uses bZip2 (bz2) rather than the usual run-length encoder (RLE) and huffman tree. The algorithm incorporates the use of Structural Similarity Index Measurement (SSIM) to discard redundant coefficients to the human eye.

See SETUP section on how to get started.

# Recent Additions

*6.10*

You can now save the image as a compressed JPEG. The JPEG won't be as small as a bz2 file but will be smaller
than the original JPEG. This compressor should also outperform Google's Guetzli. The compressed JPEG will allow you to
transfer JPEG files quicker and using less bandwidth. The compressed JPEG is also compatible with all OS. Disclaimer:
compressed JPEG file contains NO metadata. You can delete the original file at your own discretion.

*6.28*

You can run compression and decompression on entire folders. Use compressfolder.py and decompressfolder.py.

*7.22*

Updated to use parallel processing for improved runtimes.

# Setup

Check dependencies at bottom of README

__*For compress.py example usage:*__

python compress.py [-i | Full image path with file and extension] [-c | Path to folder where to save compressed file/default=working directory]
 
__*For decompress.py example usage:*__

python decompress.py [-i | use letter y to save image as jpeg/default=letter n to denote that save the image as a png] [-c | Path to compressed file with extension (.bz2)] [-d | Save the image to specified folder/default=working directory]

__*For compressfolder.py example usage:*__

python compressfolder.py [-i | folder of images] [-c | where to store compressed files]

__*For decompressfolder.py example usage:*__

python decompressfolder.py [-c | folder of compressed images] [-d | directory to store decompressed images for viewing]

__Use -h for full details.__

# Explanations

All file data is saved within the bz2 file. The PNG file that is saved when decompressed is for __VIEWING PURPOSES__.
You can delete the PNG file and original after viewing, all image data is in the bz2 file.
When comparing file sizes, compare the bz2 file to the original image (JPEG file).

# Deleting the original image is not recommended

Though it is mentioned above, it is not recommended. There is __NO__ meta data in the bz2 file.
Furthermore, this algorithm is theoretical, 1. there may
be bugs and 2. no OS can unwrap a bz2 file and decompress automatically into a viewable image. You need to use the provided decompressor to decode the specific sequence of values.

Like mentioned, this is a completely novel compression algorithm.

# Future Works

Using DNN to compress files smaller by outputting optimal quantization tables.

# Dependencies & Purposes:
 - imageio (for opening image files and saving)
 - tqdm (make UI layout nicer with progess bars)
 - scipy (running JPEG's DCT {2d})
 - numpy (makes matrix mathematics easier)
 - time (timer for length of algorithm)
 - scikit-image (SSIM metric quality)
