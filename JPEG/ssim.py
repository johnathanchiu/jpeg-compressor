# import numpy as np
#
#
# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#
#     # the lower the error, the more "similar" the two images are
#     return err