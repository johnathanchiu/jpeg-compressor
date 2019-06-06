# JPEG-COMPRESSOR

This algorithm is a theoretical modification to the preexisting JPEG algorithm for reduced file sizes. The lossless
entropy algorithm uses bZip2 (bz2) rather than the usual run-length encoder (RLE) and huffman tree.

Use compress.py to compress photos. Put in entire file path when prompted.

Use decompress.py to decompress photos. Save it as a PNG to view or view immediately.

# Setup

All file paths are currently under my own directories, you can change around the file paths to make experimenting with the compressor easier. For example changing the "root_path" variable to your own path you want to file to be saved to. These files are also configured under a special folder I put them in. Check main method to see how to fix.

# Explanations

All file data is saved within the bz2 file. I am working on developing a wrapper for the bz2 file so the image does
not have to be saved into a PNG. The PNG file that is saved when decompressed is for viewing purposes.
You can delete the PNG file and original after viewing, all image data is in the bz2 file.

# Deleting the original image is not recommended

It is not recommended, there is no meta data in the bz2 file. Furthermore, this algorithm is theoretical, 1. there may
be bugs and 2. no OS can unwrap a bz2 file and decompress automatically into a viewable image. Like I mentioned, this is a completely novel compression algorithm.

# Dependencies:
 - imageio
 - tqdm
 - scipy
 - numpy
 - time
