from skimage.measure._structural_similarity import compare_ssim as ssim
import imageio


original_sample_url = '/Users/johnathanchiu/Documents/CompressionPics/tests/IMG_0846.jpeg'
index_url = './IMG_0846.jpg'

original_sample, index = imageio.imread(original_sample_url), imageio.imread(index_url)

metric = ssim(original_sample, index, data_range=index.max() - index.min(),  multichannel=True)

print('Result SSIM Metric: ', metric)
