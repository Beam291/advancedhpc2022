from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

# file = '../img/animeImg.jpg'
file = '../img/labImg.jpg'
filePath = file

filterList = [
[0, 0, 1, 2, 1, 0, 0],
[0, 3, 13, 22, 13, 3, 0],
[1, 13, 59, 97, 59, 13, 1],
[2, 22, 97, 159, 97, 22, 2],
[1, 13, 59, 97, 59, 13, 1],
[0, 3, 13, 22, 13, 3, 0],
[0, 0, 1, 2, 1, 0, 0]]

filterArr = np.array(filterList)

def conv2D(colorArray : np.ndarray, kernel : np.ndarray, average = False):
  c_row, c_col = colorArray.shape
  k_row, k_col = kernel.shape
  
  output = np.zeros(colorArray.shape)

  pad_height = int((k_row - 1)/2)
  pad_width = int((k_col -1)/2)
  
  padded_image = np.zeros((c_row + (2 * pad_height), c_col + (2 * pad_width)))
  padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = colorArray
  
  for row in range(c_row):
    for col in range(c_col):
      output[row, col] = np.sum(kernel * padded_image[row:row + k_row, col: col + k_col])
      if average:
        output[row, col] /= kernel.shape[0] * kernel.shape[1]
  return output

def normalize(im):
   min, max = im.min(), im.max()
   return (im.astype(float)-min)/(max-min)

def convCPU(image : np.ndarray, kernel : np.ndarray):
  b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
  n_b = conv2D(b, kernel, average=True)
  n_g = conv2D(g, kernel, average=True)
  n_r = conv2D(r, kernel, average=True)
    
  nb = normalize(n_b) * 255.999
  ng = normalize(n_g) * 255.999
  nr = normalize(n_r) * 255.999
  
  result = np.dstack((nb,ng,nr)).astype(np.uint8)
  return result

@cuda.jit
def grayScale2D_GPU(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
  g = np.uint8((src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2])/ 3)
  dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g
  
def grayScale_noGPU(imgArray : np.ndarray):
  for i in range(imgArray.shape[0]):
    for j in range(imgArray.shape[1]):
      gray = np.uint8(int((imgArray[i, j, 0]) + int(imgArray[i, j, 1]) + int(imgArray[i, j, 2]))/3)
      imgArray[i, j, 0] = imgArray[i, j, 1] = imgArray[i, j, 2] = gray
  return imgArray


img = mpimg.imread(filePath)
imgShape = np.shape(img)
height, width  = imgShape[0], imgShape[1]

newImg = convCPU(img, filterArr)
plt.imshow(newImg)
plt.show()