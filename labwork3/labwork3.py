from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda, jit
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

# def grayScale(src, dst):
#     tidx = cuda.threadIdx.x + cuda.blockIdx * cuda.blockDim.x
#     g = np.uint8(src[tidx, 0] + src[tidx, 1] + src[tidx, 2] / 3)
#     dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g
    
def grayScale_noGPU(imgArray : np.ndarray):
  for i in imgArray:
    gray = ((i[0]+i[1]+[2])/3)
    i[0], i[1], i[2] = gray, gray, gray
  imgGray = imgArray.reshape(width, height, 3)
  return imgGray

@jit(target_backend ='cuda')
def grayScale_GPU(imgArray : np.ndarray):
  for i in imgArray:
    gray = ((i[0]+i[1]+[2])/3)
    i[0], i[1], i[2] = gray, gray, gray
  imgGray = imgArray.reshape(width, height, 3)
  return imgGray
    
fileName = 'labimg.jpg'
# fileName = '102368197_p0.jpg'

filePath = "img/"+fileName

img = mpimg.imread(filePath)
imgShape = np.shape(img)
width, height = imgShape[0], imgShape[1]

pixelCount = width * height
blockSize = 64
gridSize  = pixelCount / blockSize
gridSize = round(gridSize)

# hostInput = np.zeros((height, width, 3), np.uint8)
# devOutput = cuda.device_array((height, width, 3), np.uint8)
# devData = cuda.to_device(hostInput)

flatten = img.flatten().reshape(pixelCount,3)

start = timer()
imggray1 = grayScale_noGPU(flatten)
print("Without GPU: ", timer() - start)

start = timer()
imggray2 = grayScale_GPU(flatten)
print("With GPU: ", timer() - start)

