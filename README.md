# JPEG-COMPRESSOR

This algorithm is a theoretical modification to the preexisting JPEG algorithm for reduced file sizes. The lossless
entropy algorithm uses bZip2 (bz2) rather than the usual run-length encoder (RLE) and huffman tree. The algorithm incorporates the use of Structural Similarity Index Measurement (SSIM) to discard redundant coefficients to the human eye.

Use compress.py to compress photos.

Use decompress.py to decompress photos. Save it as a decompressed PNG or JPEG.

# Recent Additions

You can now save the image as a compressed JPEG. The JPEG won't be as small as a bz2 file but will be smaller
than the original JPEG. This compressor should also outperform Google's Guetzli. The compressed JPEG will allow you to
transfer JPEG files quicker and using less bandwidth. The compressed JPEG is also compatible with all OS. Disclaimer:
compressed JPEG file contains NO metadata. You can delete the original file at your own discretion.

# Setup

Check dependencies at bottom of README

For compress.py example usage:

python compress.py [-i | Full image path with file and extension] [-c | Path to folder where to save compressed file/default=working directory]
 
For decompress.py example usage:

python decompress.py [-i | use letter y to save image as jpeg/default=letter n to denote that save the image as a png] [-c | Path to compressed file with extension (.bz2)] [-d | Save the image to specified folder/default=working directory]

For compressfolder.py example usage:

python compressfolder.py [-i | folder of images] [-c | where to store compressed files]

For decompressfolder.py example usage:

python decompressfolder.py [-c | folder of compressed images] [-d | directory to store decompressed images for viewing]

Use -h for full details.

# Explanations

All file data is saved within the bz2 file. The PNG file that is saved when decompressed is for viewing purposes (emphasized).
You can delete the PNG file and original after viewing, all image data is in the bz2 file.
When comparing file sizes, compare the bz2 file to the original image (JPEG file).

# Deleting the original image is not recommended

Though it is mentioned above, it is not recommended. There is no meta data in the bz2 file.
Furthermore, this algorithm is theoretical, 1. there may
be bugs and 2. no OS can unwrap a bz2 file and decompress automatically into a viewable image. You need to use the provided decompressor to decode the specific sequence of values.

Like I mentioned, this is a completely novel compression algorithm.

# Future Works

Novel lossless entropy encoder which outperforms bz2 and gzip to be added as replacement of bz2 also coming soon.
Paper to how the entire algorithm works will be linked soon!

# Dependencies & Purposes:
 - imageio (for opening image files and saving)
 - tqdm (make UI layout nicer with progess bars)
 - scipy (running JPEG's DCT {2d})
 - numpy (makes matrix mathematics easier)
 - time (timer for length of algorithm)
 - scikit-image (SSIM metric quality)
