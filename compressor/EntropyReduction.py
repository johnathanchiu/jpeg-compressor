import numpy as np

import bz2

import struct
import os


class EntropyReduction:
    @staticmethod
    def bz2(compressed, output_file):
        def bz2_comp(output, output_file):
            filename = output_file + ".bz2"
            with bz2.open(filename, "wb") as f:
                output = bz2.compress(output, 9)
                f.write(output)
            return os.path.getsize(filename), filename

        compressed = (struct.pack('b' * len(compressed), *compressed))
        size, filename = bz2_comp(compressed, output_file)
        return size, filename

    @staticmethod
    def bz2_unc(file_name):
        def bz2_decomp(file):
            with bz2.open(file, "rb") as f:
                f_content = f.read()
                f_content = bz2.decompress(f_content)
            return f_content

        result_bytes = bz2_decomp(file_name)
        fmt = "%db" % len(result_bytes)
        result_bytes = np.asarray(list(struct.unpack(fmt, result_bytes)))
        return result_bytes
