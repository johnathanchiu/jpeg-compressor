# JPEG-COMPRESSOR

This algorithm is a theoretical modification to the preexisting JPEG algorithm for reduced file sizes. The lossless
entropy algorithm uses bZip2 (bz2) rather than the usual run-length encoder (RLE) and huffman tree.

Use compress.py to compress photos. Put in entire file path when prompted.

Use decompress.py to decompress photos. Save it as a PNG to view or otherwise view immediately using imageio.

# Recent Additions

You can now save the image as a compressed JPEG. The JPEG won't be as small as a bz2 file but will be smaller
than the original JPEG. This compressor should also outperform Google's Guetzli. The compressed JPEG will allow you to
transfer JPEG files quicker and using less bandwidth. The compressed JPEG is also compatible with all OS. Disclaimer:
compressed JPEG file contains NO metadata. You can delete the original file at your own discretion.

# Setup

All file paths are currently under my own directories, you can change around the file paths to make experimenting with
the compressor easier. For example changing the "root_path" variable to your own path you want to file to be saved to.
These files are also configured under a special folder I put them in. Check main method to see how to fix.
For compress.py change line 107, your "root_path". For decompress.py change line 83, your "root_path".

# Explanations

All file data is saved within the bz2 file. I am working on developing a wrapper for the bz2 file so the image does
not have to be saved into a PNG. The PNG file that is saved when decompressed is for viewing purposes (emphasized).
You can delete the PNG file and original after viewing, all image data is in the bz2 file.
When comparing file sizes, compare the bz2 file to the original image (JPEG file).

# Deleting the original image is not recommended

Though it is mentioned above, it is not recommended. There is no meta data in the bz2 file.
Furthermore, this algorithm is theoretical, 1. there may
be bugs and 2. no OS can unwrap a bz2 file and decompress automatically into a viewable image.
Like I mentioned, this is a completely novel compression algorithm.

# Future Works

Currently, this program only accepts one quality value. Future plans are to enable the user to choose the quality of the photo.
The algorithm will work using an SSIM metric that gives comparisons on a small partition of the image (Update: This part has been updated,
the algorithm only accepts one quality but it chooses the best possible quality in relative to the best compression ratios).
Novel lossless entropy encoder which outperforms bz2 and gzip to be added as replacement of bz2 also coming soon.
Paper to how the entire algorithm works will be linked soon!

# Dependencies & Purposes:
 - imageio (for opening image files and saving)
 - tqdm (make UI layout nicer with progess bars)
 - scipy (running JPEG's DCT {2d})
 - numpy (makes matrix mathematics easier)
 - time (timer for length of algorithm)
 - scikit-image (SSIM metric quality)
