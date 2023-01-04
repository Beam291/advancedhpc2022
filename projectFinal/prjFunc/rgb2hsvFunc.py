from numba import cuda

@cuda.jit
def RGB2HSV(src, hsvOutput):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y 
        
    R = src[tidx, tidy, 2]
    G = src[tidx, tidy, 1]
    B = src[tidx, tidy, 0]
    
    tidMax = max(R, G, B)
    tidMin = min(R, G, B)
    
    delta = tidMax - tidMin
    
    if delta == 0:
        hsvOutput[tidx, tidy, :] = 0
    elif R == tidMax:
        hsvOutput[tidx, tidy, 0] = (((G-B)/delta)%6) * 60 
    elif G == tidMax:
        hsvOutput[tidx, tidy, 0] = (((B-R)/delta)+2) * 60
    elif B == tidMax:
        hsvOutput[tidx, tidy, 0] = (((R-G)/delta)+4) * 60
     
    if tidMax == 0:
        hsvOutput[tidx, tidy, 1] = 0
    else:
        hsvOutput[tidx, tidy, 1] = delta/tidMax
    
    hsvOutput[tidx, tidy, 2] = tidMax