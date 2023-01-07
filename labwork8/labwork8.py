from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings("ignore")

file = '../img/animeImg.jpg'
# file = '../img/cute-cat.jpg'

class preImgFunc:
    def __init__(self, filePath : str):
        self.filePath = filePath

    def readImg(self):
        img = mpimg.imread(self.filePath)
        imgShape = np.shape(img)
        height, width = imgShape[0], imgShape[1]
        return img, imgShape, height, width

@cuda.jit
def RGB2HSV(src, hsvOutput):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
        
    R = src[tidx, tidy, 2]/255
    G = src[tidx, tidy, 1]/255
    B = src[tidx, tidy, 0]/255
    
    tidMax = max(R, G, B)
    tidMin = min(R, G, B)
    
    delta = tidMax - tidMin
    
    if delta == 0:
        hsvOutput[tidx, tidy, :] = 0
    elif R == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((G-B)/delta)%6) * 60) % 360
    elif G == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((B-R)/delta)+2) * 60) % 360
    elif B == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((R-G)/delta)+4) * 60) % 360
     
    if tidMax == 0:
        hsvOutput[tidx, tidy, 1] = 0
    else:
        hsvOutput[tidx, tidy, 1] = (delta/tidMax) * 100
    
    hsvOutput[tidx, tidy, 2] = tidMax*100
    
animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()
hArr = height * width

devRGBInput = cuda.to_device(img)
devHSVOutput = cuda.device_array(imgShape, np.uint8)

blockSize = (32,32)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))

start = timer()
RGB2HSV[gridSize,blockSize](devRGBInput, devHSVOutput)
print("Time: ", timer() - start)

hsvImg = devHSVOutput.copy_to_host()
imgShow = plt.imshow(hsvImg)
plt.show()
