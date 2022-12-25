from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings("ignore")

file = '../img/animeImg.jpg'
file1 = '../img/animeImg1.jpg'

@cuda.jit
def grayScale_GPU(src, src1, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    dst[tidx, tidy, 0] = (src[tidx, tidy, 0] + src1[tidx, tidy, 0]) / 2
    dst[tidx, tidy, 1] = (src[tidx, tidy, 1] + src1[tidx, tidy, 1]) / 2
    dst[tidx, tidy, 2] = (src[tidx, tidy, 2] + src1[tidx, tidy, 2]) / 2

img = mpimg.imread(file)
img1 = mpimg.imread(file1)
imgShape = np.shape(img)
width, height = imgShape[0], imgShape[1]

pixelCount = width * height
blockSize = (32,32)
gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))

devOutput = cuda.device_array(imgShape, np.uint8)
devData1 = cuda.to_device(img1)
devData = cuda.to_device(img)

start = timer()
grayScale_GPU[gridSize, blockSize](devData, devData1, devOutput)
print("With GPU: ", timer() - start)

grayImage2 = devOutput.copy_to_host()
grayImage2 = grayImage2.reshape(width, height, 3)

plt.imshow(grayImage2)
plt.show()