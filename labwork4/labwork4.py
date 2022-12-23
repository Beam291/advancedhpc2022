from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings("ignore")

# file = 'animeImg.jpg'
file = 'labImg.jpg'
filePath = 'img/' + file

@cuda.jit
def grayScale_GPU(src, dst):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
  gx = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
  gy = np.uint8((src[tidy, 0] + src[tidy, 1] + src[tidy, 2]) / 3)
  dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = gx
  dst[tidy, 0] = dst[tidy, 1] = dst[tidy, 2] = gy
  
def grayScale_noGPU(imgArray : np.ndarray):
  for i in imgArray:
    gray = np.uint8((int(i[0])+int(i[1])+int(i[2]))/3)
    i[0] = i[1] = i[2] = gray
  imgGray = imgArray.reshape(width, height, 3)
  return imgGray

img = mpimg.imread(filePath)
imgShape = np.shape(img)
width, height = imgShape[0], imgShape[1]

pixelCount = width * height
blockSize = (7,7)
gridSize = tuple(math.ceil(pixelCount/i) for i in blockSize)

# print(img.shape)

flatten = img.flatten().reshape(img.shape[0], (img.shape[1]*img.shape[2]))
print((flatten))
# flattenShape = np.shape(flatten)

# image2D = img.reshape()

# devOutput = cuda.device_array(flattenShape, np.uint8)
# devData = cuda.to_device(flatten)

# start = timer()
# grayImage1 = grayScale_noGPU(flatten)
# print("Wihout GPU: ", timer()  - start)

# start = timer()
# grayScale_GPU[gridSize, blockSize](devData, devOutput)
# print("With GPU: ", timer() - start)

# figure, axis = plt.subplots(2)
# figure.tight_layout(pad=5.0)

# grayImage2 = devOutput.copy_to_host()
# grayImage2 = grayImage2.reshape(width, height, 3)

# axis[0].imshow(grayImage1)
# axis[0].set_title("Without GPU")
# axis[1].imshow(grayImage2)
# axis[1].set_title("With GPU")
# plt.show()