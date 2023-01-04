from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
import prjFunc.rgb2hsvFunc as imgp
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

@cuda.jit
def kuwaFilter(src, dst, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    
    winA = 0
    for i in range(winLen):
        for j in range(winLen):
            winA += src[2, 2]
    

animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()
winLen = 4

devRGBInput = cuda.to_device(img)
devHSVOutput = cuda.device_array(imgShape, np.uint8)
devKuwaOutput = cuda.device_array(imgShape, np.uint8)

blockSize = (32,32)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))

# start = timer()
# imgp.RGB2HSV[gridSize,blockSize](devRGBInput, devHSVOutput)
# print("RGB2HSV Time: ", timer() - start)
b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]

devBInput = cuda.to_device(np.ascontiguousarray(b))
kuwaFilter[gridSize, blockSize](devBInput, devKuwaOutput, height, width, winLen)

# hsvImg = devHSVOutput.copy_to_host()
# vArr = hsvImg[:, :, 2]

