from numba import cuda
import numpy as np
import math

def winDow(src, vArr, tidx, tidy, winLen, height, width):
        winA = np.zeros((winLen, winLen))
        winAV = np.zeros((winLen, winLen))
        for i in range(winLen):
            for j in range(winLen):
                xValue = tidx - i
                yValue = tidy - j
                if xValue < 0 or yValue < 0:
                    continue
                else:
                    winAV[i,j] = vArr[xValue, yValue]
                    winA[i,j] = src[xValue, yValue]
        meanA = np.mean(winA)
        stanA = np.std(winAV)
        
        winB = np.zeros((winLen, winLen))
        winBV = np.zeros((winLen, winLen))
        for i in range(winLen):
            for j in range(winLen):
                xValue = tidx - i
                yValue = tidy + j
                if xValue < 0 or yValue >= width:
                    continue
                else:
                    winBV[i,j] = vArr[xValue, yValue]
                    winB[i,j] = src[xValue, yValue]
        meanB = np.mean(winB)
        stanB = np.std(winBV)
        
        winC = np.zeros((winLen, winLen))
        winCV = np.zeros((winLen, winLen))
        for i in range(winLen):
            for j in range(winLen):
                xValue = tidx + i
                yValue = tidy - j
                if xValue >= height or yValue < 0:
                    continue
                else:
                    winCV[i,j] = vArr[xValue, yValue]
                    winC[i,j] = src[xValue, yValue]
        meanC = np.mean(winC)
        stanC = np.std(winCV)
        
        winD = np.zeros((winLen, winLen))
        winDV = np.zeros((winLen, winLen))
        for i in range(winLen):
            for j in range(winLen):
                xValue = tidx + i
                yValue = tidy + j
                if xValue >= height or yValue >= width:
                    continue
                else:
                    winDV[i,j] = vArr[xValue, yValue]
                    winD[i,j] = src[xValue, yValue]
        meanD = np.mean(winD)
        stanD = np.std(winDV)
        
        minWin = min(stanA, stanB, stanC, stanD)
        
        if minWin == stanA:
            return  meanA
        elif minWin == stanB:
            return meanB
        elif minWin == stanC:
            return meanC
        elif minWin == stanD:
            return meanD

def kuwaFilter_CPU(src, imgShape, vArr, height, width, winLen):
    
    b : np.ndarray
    g : np.ndarray
    r : np.ndarray 
    b, g, r = (src[:,:,0]), (src[:,:,1]), (src[:,:,2])
    
    n_b = np.zeros(b.shape, np.uint8)
    n_g = np.zeros(g.shape, np.uint8)
    n_r = np.zeros(r.shape, np.uint8)

    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            n_b[x, y] = winDow(b, vArr, x, y, winLen, height, width)

    for x in range(g.shape[0]):
        for y in range(g.shape[1]):
            n_g[x, y] = winDow(g, vArr, x, y, winLen, height, width)
    
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            n_r[x, y] = winDow(r, vArr, x, y, winLen, height, width)

    n_img = np.dstack((n_b, n_g, n_r))
    
    return n_img

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
            if xValue < 0 or yValue >= width:
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
            if xValue >= height or yValue < 0:
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
            if xValue >= height or yValue >= width:
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
            if xValue < 0 or yValue >= width:
                winVBSDSumPow += pow((0 - meanVB),2)
            else:
                winVBSDSumPow += pow((vArr[xValue, yValue] - meanVB),2)
    
    winVCSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy - j
            if xValue >= height or yValue < 0:
                winVCSDSumPow += pow((0 - meanVC),2)
            else:
                winVCSDSumPow += pow((vArr[xValue, yValue] - meanVC),2)
    
    winVDSDSumPow = 0
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy + j
            if xValue >= height or yValue >= width:
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
def kuwaFilter_GPU_WithMemory(src, dst, vArr, height, width, winLen):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    a = 4
    
    winA = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    winAV = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    
    winB = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    winBV = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    
    winC = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    winCV = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    
    winD = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    winDV = cuda.shared.array(shape = (a,a), dtype= np.uint8)
    
    def calWin(arr, arr1, winLen):
        winSum = 0
        winVSum = 0
        for i in range(winLen):
            for j in range(winLen):
                winSum += arr[i,j]
                winVSum += arr1[i,j]
        meanW = winSum/(winLen * winLen)
        meanV = winVSum/(winLen * winLen)
        
        winVSDSum = 0
        for i in range(winLen):
            for j in range(winLen):
                winVSDSum += pow((arr1[i,j] - meanV),2)
        
        stanV = math.sqrt(winVSDSum/((winLen * winLen)))
                
        return meanW, stanV
    
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy - j
            if xValue < 0 or yValue < 0:
                continue
            else:
                winA[i,j] = src[xValue, yValue]
                winAV[i,j] = vArr[xValue, yValue]
                cuda.syncthreads()
    
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx - i
            yValue = tidy + j
            if xValue < 0 or yValue >= width:
                continue
            else:
                winB[i,j] = src[xValue, yValue]
                winBV[i,j] = vArr[xValue, yValue]
                cuda.syncthreads()
            
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy - j
            if xValue >= height or yValue < 0:
                continue
            else:
                winC[i,j] = src[xValue, yValue]
                winCV[i,j] = vArr[xValue, yValue]
                cuda.syncthreads()
            
    for i in range(winLen):
        for j in range(winLen):
            xValue = tidx + i
            yValue = tidy + j
            if xValue >= height or yValue >= width:
                continue
            else:
                winD[i,j] = src[xValue, yValue]
                winDV[i,j] = vArr[xValue, yValue]
                cuda.syncthreads()
            
    meanA, stanA = calWin(winA, winAV, winLen)
    meanB, stanB = calWin(winB, winBV, winLen)
    meanC, stanC = calWin(winC, winCV, winLen)
    meanD, stanD = calWin(winD, winDV, winLen)
    
    minWin = min(stanA, stanB, stanC, stanD)
    
    if minWin == stanA:
        dst[tidx, tidy] = meanA
    elif minWin == stanB:
        dst[tidx, tidy] = meanB
    elif minWin == stanC:
        dst[tidx, tidy] = meanC
    elif minWin == stanD:
        dst[tidx, tidy] = meanD