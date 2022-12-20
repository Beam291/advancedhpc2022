from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from numba import cuda, jit
from timeit import default_timer as timer


# def grayScale(src, dst):
    # tidx = cuda.threadIdx.x + cuda.blockIdx * cuda.blockDim.x
    # g = np.uint8(src[tidx, 0] + src[tidx, 1] + src[tidx, 2] / 3)
    # dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g
    
def grayScale_noGPU(imgArray : np.ndarray):
  for i in imgArray:
    gray = ((i[0]+i[1]+[2])/3)
    i[0], i[1], i[2] = gray, gray, gray
  imgGray = imgArray.reshape(width, height, 3)
  return imgGray

@jit(target_backend='cuda')
def grayScale_GPU(imgArray : np.ndarray):
  for i in imgArray:
    gray = ((i[0]+i[1]+[2])/3)
    i[0], i[1], i[2] = gray, gray, gray
  imgGray = imgArray.reshape(width, height, 3)
  return imgGray
    
fileName = 'labimg.jpg'

filePath = "img/"+fileName

im = Image.open(filePath)
width, height = im.size
im.close()

hostInput = np.zeros((height, width, 3), np.uint8)
devOutput = cuda.device_array((height, width, 3), np.uint8)
devData = cuda.to_device(hostInput)

pixelCount = width * height
blockSize = 64
gridSize  = pixelCount / blockSize

img = mpimg.imread(filePath)

flatten = img.flatten().reshape(pixelCount,3)

start = timer()
imggray1 = grayScale_noGPU(flatten)
# print("Without GPU: ", timer() - start())

start = timer()
imggray2 = grayScale_GPU(flatten)
# print("With GPU: ", timer() - start())


# print(img)
# print(flatten)
# imgplot = plt.imshow(devInput)
# plt.show()

# t1 = time.time()


# for i in flatten:
#   gray = ((i[0]+i[1]+[2])/3)
#   i[0], i[1], i[2] = gray, gray, gray

# t2 = time.time()

# print(t2-t1)

# imgray1 = flatten.reshape(width, height, 3)
# imgplot = plt.imshow(imgray1)
# plt.show()

# print(grayScale[gridSize, blockSize](devInput, devOutput))