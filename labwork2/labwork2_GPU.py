from numba import cuda
from numba.cuda.cudadrv import enums
from pynvml import *

cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128,
}

device = cuda.get_current_device()
attribs= [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
# for attr in attribs:
#     print(attr, '=', getattr(device, attr))
    
deviceName = device.name
print("Device name: ", deviceName)
print("Multiprocessor: ", getattr(device, 'MULTIPROCESSOR_COUNT'))
print("Clock rate: ", getattr(device, 'CLOCK_RATE'))

my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
my_cc = device.compute_capability
cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
total_cores = cores_per_sm*my_sms
print("GPU compute capability: " , my_cc)
print("GPU total number of SMs: " , my_sms)
print("total cores: " , total_cores)

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total} Byte')
print(f'free     : {info.free} Byte')
print(f'used     : {info.used} Byte')