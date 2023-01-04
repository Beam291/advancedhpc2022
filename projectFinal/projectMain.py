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
def kuwaFilter(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    
    vValue = vArr[tidx, tidy]
    
    winASum = 0
    winASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy - i
            if xValue < 0 or yValue < 0:
                winASDSum += pow((255 - vValue), 2) 
                winASum += 0
            else:
                winASDSum += pow((src[xValue, yValue] - vValue), 2) 
                winASum += src[xValue, yValue]
                
    winBSum = 0
    winBSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy - i
            if xValue > width or yValue < 0:
                winBSDSum += pow((255 - vValue), 2) 
                winBSum += 0
            else:
                winBSDSum += pow((src[xValue, yValue] - vValue), 2) 
                winBSum += src[xValue, yValue]
    
    winCSum = 0
    winCSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy + i
            if xValue < 0 or yValue > height:
                winCSDSum += pow((255 - vValue), 2) 
                winCSum += 0
            else:
                winCSDSum += pow((src[xValue, yValue] - vValue), 2) 
                winCSum += src[xValue, yValue]
    
    winDSum = 0
    winDSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy + i
            if xValue > width or yValue > height:
                winDSDSum += pow((255 - vValue), 2) 
                winDSum += 0
            else:
                winDSDSum += pow((src[xValue, yValue] - vValue), 2) 
                winDSum += src[xValue, yValue]
    
    stanA = math.sqrt(winASDSum/15)
    stanB = math.sqrt(winBSDSum/15)
    stanC = math.sqrt(winCSDSum/15)
    stanD = math.sqrt(winDSDSum/15)
    
    minWin = min(stanA, stanB, stanC, stanD)
    
    if minWin == stanA:
        dst[tidx, tidy] = (winASum/16)
    elif minWin == stanB:
        dst[tidx, tidy] = (winBSum/16)
    elif minWin == stanC:
        dst[tidx, tidy] = (winCSum/16)
    elif minWin == stanD:
        dst[tidx, tidy] = (winDSum/16)
    

animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()
winLen = 4

devRGBInput = cuda.to_device(img)
devHSVOutput = cuda.device_array(imgShape, np.uint8)
devKuwaOutput = cuda.device_array(imgShape, np.uint8)

blockSize = (32,32)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))

start = timer()
imgp.RGB2HSV[gridSize,blockSize](devRGBInput, devHSVOutput)
print("RGB2HSV Time: ", timer() - start)

hsvImg = devHSVOutput.copy_to_host()
vArr = np.ascontiguousarray(hsvImg[:,:,2])
vArrInput = cuda.to_device(vArr)

b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]

devBInput = cuda.to_device(np.ascontiguousarray(b))
devBOutput = cuda.device_array((height, width), np.uint8)
kuwaFilter[gridSize, blockSize](devBInput, devBOutput, vArrInput, height, width, winLen)
n_b = devBOutput.copy_to_host()

devGInput = cuda.to_device(np.ascontiguousarray(g))
devGOutput = cuda.device_array((height, width), np.uint8)
kuwaFilter[gridSize, blockSize](devGInput, devGOutput, vArrInput, height, width, winLen)
n_g = devGOutput.copy_to_host()

devRInput = cuda.to_device(np.ascontiguousarray(r))
devROutput = cuda.device_array((height, width), np.uint8)
kuwaFilter[gridSize, blockSize](devRInput, devROutput, vArrInput, height, width, winLen)
n_r = devROutput.copy_to_host()

kuwaImg = np.dstack((n_b, n_g, n_r))

plt.imshow(kuwaImg)
plt.show()
# hsvImg = devHSVOutput.copy_to_host()
# vArr = hsvImg[:, :, 2]

