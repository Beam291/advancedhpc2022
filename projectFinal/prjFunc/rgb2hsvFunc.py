from numba import cuda

@cuda.jit
def RGB2HSV(src, hsvOutput):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
        
    R = src[tidx, tidy, 2]/255
    G = src[tidx, tidy, 1]/255
    B = src[tidx, tidy, 0]/255
    
    tidMax = max(R, G, B)
    tidMin = min(R, G, B)
    
    delta = tidMax - tidMin
    
    if delta == 0:
        hsvOutput[tidx, tidy, :] = 0
    elif R == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((G-B)/delta)%6) * 60) % 360
    elif G == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((B-R)/delta)+2) * 60) % 360
    elif B == tidMax:
        hsvOutput[tidx, tidy, 0] = ((((R-G)/delta)+4) * 60) % 360
     
    if tidMax == 0:
        hsvOutput[tidx, tidy, 1] = 0
    else:
        hsvOutput[tidx, tidy, 1] = (delta/tidMax) * 100
    
    hsvOutput[tidx, tidy, 2] = tidMax*100