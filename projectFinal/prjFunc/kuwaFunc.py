from numba import cuda
import math

@cuda.jit
def kuwaFilter_GPU(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
    
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
      
@cuda.jit
def kuwaFilter_GPU_WithoutMemory(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    def winDow(src, vArr, c_x, c_y, n_x, n_y , height, width, winLen):
        winSum = 0
        winVSum = 0
        population = winLen * winLen

        for i in range(c_x, n_x):
            for j in range(c_y, n_y):
                if i < 0 or i > width or j < 0 or j > height:
                    winSum += 255
                    winVSum += 100
                else:
                    winSum += src[i, j]
                    winVSum += vArr[i,j]

        winMean = winSum/population
        winVMean = winVSum/population
        
        winVSDSum = 0
        for i in range(c_x, n_x):
            for j in range(c_y, n_y):
                if i < 0 or i > width or j < 0 or j > height:
                    winVSDSum += pow((0 - winVMean), 2)
                else:
                    winVSDSum += pow((vArr[i, j] - winVMean), 2)
        
        winVSD = math.sqrt(winVSDSum/population)

        return winMean, winVSD
    
    winAmean, winVASD = winDow(src, vArr, tidx, tidy, tidx - winLen, tidy - winLen, width, height, winLen)
    winBmean, winVBSD = winDow(src, vArr, tidx, tidy, tidx + winLen, tidy - winLen, width, height, winLen)
    winCmean, winVCSD = winDow(src, vArr, tidx, tidy, tidx - winLen, tidy + winLen, width, height, winLen)
    winDmean, winVDSD = winDow(src, vArr, tidx, tidy, tidx + winLen, tidy + winLen, width, height, winLen)

    minWin = min(winVASD, winVBSD, winVCSD, winVDSD)

    dst[tidx, tidy] = winAmean
    
    # if minWin == winVASD:
    #     dst[tidx, tidy] = winAmean
    # elif minWin == winVBSD:
    #     dst[tidx, tidy] = winBmean
    # elif minWin == winVCSD:
    #     dst[tidx, tidy] = winCmean
    # elif minWin == winVDSD:
    #     dst[tidx, tidy] = winDmean
        
@cuda.jit
def kuwaFilter_GPU_WithMemory(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

