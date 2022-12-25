from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings("ignore")

file = '../img/animeImg.jpg'
# file = '../img/labImg.jpg'
filePath =  file

@cuda.jit
def grayScale_GPU(src, dst, gamma):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    def newGamma(i, gamma):
        n_g = i + gamma
        if n_g > 255:
            n_g = 255
        elif n_g < 0:
            n_g = 0
        return n_g
    
    dst[tidx, tidy, 0] = newGamma(src[tidx, tidy, 0], gamma)
    dst[tidx, tidy, 1] = newGamma(src[tidx, tidy, 1], gamma)
    dst[tidx, tidy, 2] = newGamma(src[tidx, tidy, 2], gamma)


img = mpimg.imread(filePath)
imgShape = np.shape(img)
width, height = imgShape[0], imgShape[1]

pixelCount = width * height
blockSize = (32,32)
gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))

devOutput = cuda.device_array(imgShape, np.uint8)
devData = cuda.to_device(img)

start = timer()
grayScale_GPU[gridSize, blockSize](devData, devOutput, -155)
print("With GPU: ", timer() - start)

grayImage2 = devOutput.copy_to_host()
grayImage2 = grayImage2.reshape(width, height, 3)

plt.imshow(grayImage2)
plt.show()