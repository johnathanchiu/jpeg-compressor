from skimage.measure._structural_similarity import compare_ssim as ssim
import imageio

''' 
args:
pass in the path of the original image in 'original_sample_url'
pass in the path of the resulting image in 'index_url'
'''
original_sample_url = '/Users/johnathanchiu/Documents/CompressionPics/tests/IMG_0846.jpeg'
index_url = './IMG_0846.jpg'

original_sample, index = imageio.imread(original_sample_url), imageio.imread(index_url)

metric = ssim(original_sample, index, data_range=index.max() - index.min(),  multichannel=True)

print('Result SSIM Metric: ', metric)
