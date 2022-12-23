from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings("ignore")

# file = 'animeImg.jpg'
file = 'labImg.jpg'
filePath = 'img/' + file

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

start = timer()
grayScale2D_GPU[gridSize, blockSize](devData, devOutput)
print("With GPU: ", timer() - start)

figure, axis = plt.subplots(2)
figure.tight_layout(pad=5.0)

grayImage2 = devOutput.copy_to_host()
grayImage2 = grayImage2.reshape(width, height, 3)

axis[0].imshow(grayImage1)
axis[0].set_title("Without GPU")
axis[1].imshow(grayImage2)
axis[1].set_title("With GPU")
plt.show()