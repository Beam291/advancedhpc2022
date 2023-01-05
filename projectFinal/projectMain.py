from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
import prjFunc.rgb2hsvFunc as imgp
import prjFunc.kuwaFunc as kuwa
warnings.filterwarnings("ignore")

file = '../img/animeImg.jpg'
# file = '../img/yuru.jpg'

class preImgFunc:
    def __init__(self, filePath : str):
        self.filePath = filePath

    def readImg(self):
        img = mpimg.imread(self.filePath)
        imgShape = np.shape(img)
        height, width = imgShape[0], imgShape[1]
        return img, imgShape, height, width
    
animeImg = preImgFunc(file)
img, imgShape, height, width = animeImg.readImg()
winLen = 5

devRGBInput = cuda.to_device(img)
devHSVOutput = cuda.device_array(imgShape, np.uint8)
devKuwaOutput = cuda.device_array(imgShape, np.uint8)

blockSize = (16,16)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))

start = timer()
imgp.RGB2HSV[gridSize,blockSize](devRGBInput, devHSVOutput)
print("RGB2HSV Time: ", timer() - start)

hsvImg = devHSVOutput.copy_to_host()
vArr = np.ascontiguousarray(hsvImg[:,:,2])
vArrInput = cuda.to_device(vArr)

b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]

# start = timer()
# devBInput = cuda.to_device(np.ascontiguousarray(b))
# devBOutput = cuda.device_array((height, width), np.uint8)
# kuwa.kuwaFilter_GPU[gridSize, blockSize](devBInput, devBOutput, vArrInput, height, width, winLen)
# n_b = devBOutput.copy_to_host()

# devGInput = cuda.to_device(np.ascontiguousarray(g))
# devGOutput = cuda.device_array((height, width), np.uint8)
# kuwa.kuwaFilter_GPU[gridSize, blockSize](devGInput, devGOutput, vArrInput, height, width, winLen)
# n_g = devGOutput.copy_to_host()

# devRInput = cuda.to_device(np.ascontiguousarray(r))
# devROutput = cuda.device_array((height, width), np.uint8)
# kuwa.kuwaFilter_GPU[gridSize, blockSize](devRInput, devROutput, vArrInput, height, width, winLen)
# n_r = devROutput.copy_to_host()
# print("Kuwahara Filter Time: ", timer() - start)

start = timer()
devBInput = cuda.to_device(np.ascontiguousarray(b))
devBOutput = cuda.device_array((height, width), np.uint8)
kuwa.kuwaFilter_GPU_WithoutMemory[gridSize, blockSize](devBInput, devBOutput, vArrInput, height, width, winLen)
n_b = devBOutput.copy_to_host()
print(n_b)

devGInput = cuda.to_device(np.ascontiguousarray(g))
devGOutput = cuda.device_array((height, width), np.uint8)
kuwa.kuwaFilter_GPU_WithoutMemory[gridSize, blockSize](devGInput, devGOutput, vArrInput, height, width, winLen)
n_g = devGOutput.copy_to_host()

devRInput = cuda.to_device(np.ascontiguousarray(r))
devROutput = cuda.device_array((height, width), np.uint8)
kuwa.kuwaFilter_GPU_WithoutMemory[gridSize, blockSize](devRInput, devROutput, vArrInput, height, width, winLen)
n_r = devROutput.copy_to_host()
print("Kuwahara Filter Time: ", timer() - start)

kuwaImg = np.dstack((n_b, n_g, n_r))

plt.imshow(kuwaImg)
plt.show()

