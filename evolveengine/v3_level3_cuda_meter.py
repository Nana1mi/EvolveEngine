"""Trusted host-only CUDA memory sampling on exactly one pinned device.

This reports CUDA-visible free and total bytes, not candidate-resident weights.
The meter is never used in the timed lane. A separate process is used during
calibration to establish whether allocations are visible across contexts.
"""
from contextlib import contextmanager
import ctypes as c
from time import monotonic_ns
from uuid import UUID


class CudaMeter:
    def __init__(self, expected_uuid):
        self.api = c.CDLL('libcuda.so.1')
        self.calls = {}
        self._call('cuInit',(c.c_uint,),0)
        count,device = c.c_int(),c.c_int()
        self._call('cuDeviceGetCount',(c.POINTER(c.c_int),),c.byref(count))
        if count.value != 1:
            raise ValueError('CUDA meter requires one visible device')
        self._call('cuDeviceGet',(c.POINTER(c.c_int),c.c_int),c.byref(device),0)
        id16 = (c.c_ubyte*16)()
        self._call('cuDeviceGetUuid_v2',(c.c_void_p,c.c_int),c.byref(id16),device)
        self.uuid = 'GPU-'+str(UUID(bytes=bytes(id16)))
        if self.uuid != expected_uuid:
            raise ValueError('CUDA meter device differs from node registration')
        driver = c.c_int()
        self._call('cuDriverGetVersion',(c.POINTER(c.c_int),),c.byref(driver))
        self.driver_api = driver.value
        self.device = device

    def _call(self, name, args, *values):
        if name not in self.calls:
            fn = getattr(self.api,name)
            fn.argtypes,fn.restype = args,c.c_int
            self.calls[name] = fn
        rc = self.calls[name](*values)
        if rc:
            raise ValueError(f'{name} failed with CUDA status {rc}')

    @contextmanager
    def context(self):
        ctx = c.c_void_p()
        self._call('cuDevicePrimaryCtxRetain',(c.POINTER(c.c_void_p),c.c_int),c.byref(ctx),self.device)
        try:
            self._call('cuCtxSetCurrent',(c.c_void_p,),ctx)
            yield self
        finally:
            try:
                self._call('cuCtxSetCurrent',(c.c_void_p,),None)
            finally:
                self._call('cuDevicePrimaryCtxRelease_v2',(c.c_int,),self.device)

    def sample(self):
        free,total = c.c_size_t(),c.c_size_t()
        self._call('cuMemGetInfo_v2',(c.POINTER(c.c_size_t),)*2,c.byref(free),c.byref(total))
        if not 0 < free.value <= total.value <= 256*(1<<30):
            raise ValueError('impossible CUDA-visible heap counts')
        return dict(at_ns=monotonic_ns(),device_uuid=self.uuid,driver_api=self.driver_api,
                    total_bytes=total.value,free_bytes=free.value,used_bytes=total.value-free.value)

    @contextmanager
    def calibration_allocation(self, size):
        if type(size) is not int or size not in (12<<20,16<<20,128<<20,512<<20):
            raise ValueError('not a registered calibration allocation')
        ptr = c.c_uint64()
        self._call('cuMemAlloc_v2',(c.POINTER(c.c_uint64),c.c_size_t),c.byref(ptr),size)
        try:
            self._call('cuMemsetD8_v2',(c.c_uint64,c.c_ubyte,c.c_size_t),ptr,0x53,size)
            self._call('cuCtxSynchronize',())
            yield
        finally:
            self._call('cuMemFree_v2',(c.c_uint64,),ptr)
