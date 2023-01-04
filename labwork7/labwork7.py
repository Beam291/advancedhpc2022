from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import PIL
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

file = '../img/animeImg.jpg'

class preImgFunc:
    def __init__(self, filePath : str):
        self.filePath = filePath

    def readImg(self):
        img = mpimg.imread(self.filePath)
        imgShape = np.shape(img)
        height, width = imgShape[0], imgShape[1]
        return img, imgShape, height, width
    
class GPUSetting:
    def __init__(self) -> None:
        pass
    
    def cudaInput(self, img : np.ndarray, shape : tuple, dataType = None):
        devInput = cuda.to_device(img)
        devOutput = cuda.device_array(shape, dtype= dataType)
        return devInput, devOutput

@cuda.jit
def grayScale_GPU( src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    g = (src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2])/ 3
    dst[tidx, tidy] = g

@cuda.jit
def grayScaleStretch_GPU(src, dst, max, min):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    n_g = ((src[tidx, tidy] - min)/(max - min)) * 255
    dst[tidx, tidy] = n_g

@cuda.reduce
def findMax(a, b):
    if a > b:
        max = a
    else:
        max = b
    return max
    
@cuda.reduce
def findMin(a, b):
    if a < b:
        min = a
    else:
        min = b
    return min

animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()

blockSize = (32,32)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))

setting = GPUSetting()
devInput, devOutput = setting.cudaInput(img, (height, width))
grayScale_GPU[gridSize, blockSize](devInput, devOutput)
grayImg = devOutput.copy_to_host()

maxIns = findMax(grayImg.flatten())
minIns = findMin(grayImg.flatten())

devInput1, devOutput1 = setting.cudaInput(grayImg, (height, width))
grayScaleStretch_GPU[gridSize, blockSize](devInput1, devOutput1, maxIns, minIns)
n_grayImg = devOutput1.copy_to_host()

plt.imshow(n_grayImg, cmap = 'gray')
plt.show()