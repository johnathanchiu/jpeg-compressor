def convertBin(num, bits=8):
    s = bin(num & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)


def convertInt(binary, bits=8):
    binary = int(binary, 2)
    """compute the 2's complement of int value val"""
    if (binary & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        binary = binary - (1 << bits)  # compute negative value
    return binary  # return positive value as is

