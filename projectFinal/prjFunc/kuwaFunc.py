from numba import cuda
import numpy as np
import math

@cuda.jit
def kuwaFilter_GPU(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    
    winASum = 0
    winVASDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy - j
            if xValue < 0 or yValue < 0:
                winVASDSum += 0
                winASum += 0
            else:
                winVASDSum += vArr[xValue, yValue]
                winASum += src[xValue, yValue]
                
    winBSum = 0
    winVBSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy + j
            if xValue < 0 or yValue > width:
                winVBSDSum += 0
                winBSum += 0
            else:
                winVBSDSum += vArr[xValue, yValue]
                winBSum += src[xValue, yValue]
    
    winCSum = 0
    winVCSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy - j
            if xValue > height or yValue < 0:
                winVCSDSum += 0
                winCSum += 0
            else:
                winVCSDSum += vArr[xValue, yValue] 
                winCSum += src[xValue, yValue]
    
    winDSum = 0
    winVDSDSum = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy + j
            if xValue > height or yValue > width:
                winVDSDSum += 0
                winDSum += 0
            else:
                winVDSDSum += vArr[xValue, yValue]
                winDSum += src[xValue, yValue]
    
    meanVA = winVASDSum/((winLen * winLen))
    meanVB = winVBSDSum/((winLen * winLen))
    meanVC = winVCSDSum/((winLen * winLen))
    meanVD = winVDSDSum/((winLen * winLen))
    
    winVASDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy - j
            if xValue < 0 or yValue < 0:
                winVASDSumPow += pow((0 - meanVA),2)
            else:
                winVASDSumPow += pow((vArr[xValue, yValue] - meanVA),2)
                
    winVBSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy + j
            if xValue < 0 or yValue > width:
                winVBSDSumPow += pow((0 - meanVB),2)
            else:
                winVBSDSumPow += pow((vArr[xValue, yValue] - meanVB),2)
    
    winVCSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy - j
            if xValue > height or yValue < 0:
                winVCSDSumPow += pow((0 - meanVC),2)
            else:
                winVCSDSumPow += pow((vArr[xValue, yValue] - meanVC),2)
    
    winVDSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy + j
            if xValue > height or yValue > width:
                winVDSDSumPow += pow((0 - meanVD),2)
            else:
                winVDSDSumPow += pow((vArr[xValue, yValue] - meanVD),2)
    
    stanA = math.sqrt(winVASDSumPow/((winLen * winLen) -1))
    stanB = math.sqrt(winVBSDSumPow/((winLen * winLen) -1))
    stanC = math.sqrt(winVCSDSumPow/((winLen * winLen) -1))
    stanD = math.sqrt(winVDSDSumPow/((winLen * winLen) -1))
    
    minWin = min(stanA, stanB, stanC, stanD)
    
    if minWin == stanA:
        dst[tidx, tidy] = (winASum/(winLen * winLen))
    elif minWin == stanB:
        dst[tidx, tidy] = (winBSum/(winLen * winLen))
    elif minWin == stanC:
        dst[tidx, tidy] = (winCSum/(winLen * winLen))
    elif minWin == stanD:
        dst[tidx, tidy] = (winDSum/(winLen * winLen))
      
# @cuda.jit
# def kuwaFilter_GPU_WithoutMemory(src, dst, vArr, height, width, winLen):
#     tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#     tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
#     def winDow(src, vArr, c_x, c_y, n_x, n_y , height, width, winLen):
#         winSum = 0
#         winVSum = 0
#         population = winLen * winLen
#         sx = 1
#         sy = 1
        
#         if n_x < 0:
#             sx = -1
#         if n_y < 0:
#             sy = -1

#         for i in range(c_x, n_x, sx):
#             for j in range(c_y, n_y, sy):
#                 if i < 0 or i > height or j < 0 or j > width:
#                     winSum += 0
#                     winVSum += 0
#                 else:
#                     winSum += src[i, j]
#                     winVSum += vArr[i,j]

#         winMean = winSum/population
#         # print(winMean)
#         winVMean = winVSum/population
        
#         winVSDSum = 0
#         for i in range(c_x, n_x, sx):
#             for j in range(c_y, n_y, sy):
#                 if i < 0 or i > height or j < 0 or j > width:
#                     winVSDSum += pow((0 - winVMean), 2)
#                 else:
#                     winVSDSum += pow((vArr[i, j] - winVMean), 2)
        
#         winVSD = math.sqrt(winVSDSum/population)

#         return winMean, winVSD
    
#     winAmean, winVASD = winDow(src, vArr, tidx, tidy, tidx - winLen, tidy - winLen, height, width, winLen)
#     winBmean, winVBSD = winDow(src, vArr, tidx, tidy, tidx - winLen, tidy + winLen, height, width, winLen)
#     winCmean, winVCSD = winDow(src, vArr, tidx, tidy, tidx + winLen, tidy - winLen, height, width, winLen)
#     winDmean, winVDSD = winDow(src, vArr, tidx, tidy, tidx + winLen, tidy + winLen, height, width, winLen)

#     minWin = min(winVASD, winVBSD, winVCSD, winVDSD)

#     dst[tidx, tidy] = winCmean
    
#     # if minWin == winVASD:
#     #     dst[tidx, tidy] = winAmean
#     # elif minWin == winVBSD:
#     #     dst[tidx, tidy] = winBmean
#     # elif minWin == winVCSD:
#     #     dst[tidx, tidy] = winCmean
#     # elif minWin == winVDSD:
#     #     dst[tidx, tidy] = winDmean
        
@cuda.jit
def kuwaFilter_GPU_WithMemory(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    winColor = cuda.shared.array(((winLen * 2) - 1, (winLen * 2) - 1), dtype= np.float64)
    winV = cuda.shared.array((winLen * 2, winLen * 2), dtype= np.float64)
    
    # for i in range((winLen * 2))