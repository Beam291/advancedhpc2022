from numba import cuda
from numba.cuda.cudadrv import enums


# print(cuda.detect())
device = cuda.get_current_device()
attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
for attr in attribs:
    print(attr, '=', getattr(device, attr))
    
print(getattr(device, 'MULTIPROCESSOR_COUNT'))
print(getattr(device, 'CLOCK_RATE'))
# print()