from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import pandas as pd
import warnings
import cv2
warnings.filterwarnings("ignore")

# file = 'animeImg.jpg'
file = 'labImg.jpg'
filePath = 'img/' + file

filterList = [
[0, 0, 1, 2, 1, 0, 0],
[0, 3, 13, 22, 13, 3, 0],
[1, 13, 59, 97, 59, 13, 1],
[2, 22, 97, 159, 97, 22, 2],
[1, 13, 59, 97, 59, 13, 1],
[0, 3, 13, 22, 13, 3, 0],
[0, 0, 1, 2, 1, 0, 0]]

filterArr = np.array(filterList)

# def convCPU(imgArray : np.ndarray, filter : np.ndarray):
#   h, w, c = imgArray.shape)
      
  # filterReshape = filter.reshape(7,7,3)
  # conv  = np.multiply(imgArray, filter)
  # Z = np.sum(conv)
  
  # return Z

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
height, width  = imgShape[0], imgShape[1]

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(image.shape)
plt.imshow(image, cmap="gray")
plt.show()
# print("Converted to Gray Channel. Size : {}".format(image.shape))

# print(filterArr)
# print(convCPU(img, filterArr, width, height))
# print(type(convCPU(img, filterArr, width, height)))
# b, g, r    = img[:, :, 0], img[:, :, 1], img[:, :, 2]
# hello = np.multiply(b, filterArr)

# print(img)
# print(imgShape)

# print(np.random.randint(2, size=(4, 3, 2)))

# pixelCount = width * height
# blockSize = (32,32)
# gridSize = (math.ceil(width/blockSize[0]), math.ceil(height/blockSize[1]))

# devOutput = cuda.device_array(imgShape, np.uint8)
# devData = cuda.to_device(img)

# start = timer()
# grayImage1 = grayScale_noGPU(img)
# print("Wihout GPU: ", timer()  - start)

# # List all the thread per block can possible be use  
# tempBlockSizeList = []
# for x in range(32):
#   for y in range(32):
#     if ((x+1) * (y+1)) <= 1024:
#       tempBlockSizeList.append([x+1, y+1])

# # Remove the thread per block that be duplicated
# blockSizeList = []
# for lst in tempBlockSizeList:
#     if sorted(lst) not in blockSizeList:
#         blockSizeList.append(lst)

# timeResult = {}
# # Start grayscale the image from each different block size 
# for i in blockSizeList:
#   blockDim = tuple(i)
#   gridDim = (math.ceil(width/blockDim[0]), math.ceil(height/blockDim[1]))
#   start = timer()
#   grayScale2D_GPU[gridSize, blockSize](devData, devOutput)
#   end = timer() - start
#   timeResult[blockDim] = end 

# timeResultDF = pd.DataFrame(timeResult.items(), columns=['blockSize', 'timeResult'])
# print('')
# print("Table of time result each time run:")
# print(timeResultDF.head())