from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# file = '../img/animeImg.jpg'
file = '../img/labImg.jpg'
filePath = file

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
width, height = imgShape[0], imgShape[1]

pixelCount = width * height
blockSize = (32,32)
gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))

devOutput = cuda.device_array(imgShape, np.uint8)
devData = cuda.to_device(img)

start = timer()
grayImage1 = grayScale_noGPU(img)
print("Wihout GPU: ", timer()  - start)

# List all the thread per block can possible be use  
tempBlockSizeList = []
for x in range(32):
  for y in range(32):
    if ((x+1) * (y+1)) <= 1024:
      tempBlockSizeList.append([x+1, y+1])

# Remove the thread per block that be duplicated
blockSizeList = []
for lst in tempBlockSizeList:
    if sorted(lst) not in blockSizeList:
        blockSizeList.append(lst)

timeResult = {}
# Start grayscale the image from each different block size 
for i in blockSizeList:
  blockDim = tuple(i)
  gridDim = (math.ceil(width/blockDim[0]), math.ceil(height/blockDim[1]))
  start = timer()
  grayScale2D_GPU[gridSize, blockSize](devData, devOutput)
  end = timer() - start
  timeResult[blockDim] = end 

timeResultDF = pd.DataFrame(timeResult.items(), columns=['blockSize', 'timeResult'])
print('')
print("Table of time result each time run:")
print(timeResultDF.head())