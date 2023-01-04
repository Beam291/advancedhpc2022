from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
import prjFunc.rgb2hsvFunc as imgp
warnings.filterwarnings("ignore")

# file = '../img/hey.jpg'
file = '../img/animeImg.jpg'
# file = '../img/animeImg2.jpg'

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
    winVASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy - i
            if xValue < 0 or yValue < 0:
                winVASDSum += 100
                winASum += 255
            else:
                winVASDSum += vArr[xValue, yValue]
                winASum += src[xValue, yValue]
                
    winBSum = 0
    winVASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy - i
            if xValue > width or yValue < 0:
                winVASDSum += 100
                winBSum += 255
            else:
                winVASDSum += vArr[xValue, yValue]
                winBSum += src[xValue, yValue]
    
    winCSum = 0
    winVASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy + i
            if xValue < 0 or yValue > height:
                winVASDSum += 100
                winCSum += 255
            else:
                winVASDSum += vArr[xValue, yValue] 
                winCSum += src[xValue, yValue]
    
    winDSum = 0
    winVASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy + i
            if xValue > width or yValue > height:
                winVASDSum += 100
                winDSum += 255
            else:
                winVASDSum += vArr[xValue, yValue]
                winDSum += src[xValue, yValue]
    
    meanVA = winVASDSum/((winLen * winLen))
    meanVB = winVASDSum/((winLen * winLen))
    meanVC = winVASDSum/((winLen * winLen))
    meanVD = winVASDSum/((winLen * winLen))
    
    winVASDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy - i
            if xValue < 0 or yValue < 0:
                winVASDSumPow += pow((0 - meanVA),2)
            else:
                winVASDSumPow += pow((vArr[xValue, yValue] - meanVA),2)
                
    winVBSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy - i
            if xValue > width or yValue < 0:
                winVBSDSumPow += pow((0 - meanVB),2)
            else:
                winVBSDSumPow += pow((vArr[xValue, yValue] - meanVB),2)
    
    winVCSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - j
            yValue = tidy + i
            if xValue < 0 or yValue > height:
                winVCSDSumPow += pow((0 - meanVC),2)
            else:
                winVCSDSumPow += pow((vArr[xValue, yValue] - meanVC),2)
    
    winVDSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + j
            yValue = tidy + i
            if xValue > width or yValue > height:
                winVDSDSumPow += pow((0 - meanVD),2)
            else:
                winVDSDSumPow += pow((vArr[xValue, yValue] - meanVD),2)
    
    stanA = math.sqrt(winVASDSumPow/((winLen * winLen)))
    stanB = math.sqrt(winVBSDSumPow/((winLen * winLen)))
    stanC = math.sqrt(winVCSDSumPow/((winLen * winLen)))
    stanD = math.sqrt(winVDSDSumPow/((winLen * winLen)))
    
    minWin = min(stanA, stanB, stanC, stanD)
    
    if minWin == stanA:
        dst[tidx, tidy] = (winASum/(winLen * winLen))
    elif minWin == stanB:
        dst[tidx, tidy] = (winBSum/(winLen * winLen))
    elif minWin == stanC:
        dst[tidx, tidy] = (winCSum/(winLen * winLen))
    elif minWin == stanD:
        dst[tidx, tidy] = (winDSum/(winLen * winLen))
    

animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()
winLen = 5

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

