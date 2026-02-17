use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

/// A typed GPU memory buffer backed by `cudarc`.
///
/// `GpuBuffer<T>` provides typed access to a contiguous region of GPU memory.
/// It does NOT own the underlying allocation when sourced from a memory pool —
/// dropping it returns the memory to the pool rather than calling `cudaFree`.
pub struct GpuBuffer<T: bytemuck::Pod> {
    inner: GpuBufferInner<T>,
    /// Cached raw device pointer (set at creation, never changes).
    raw_ptr: u64,
    len: usize,
}

enum GpuBufferInner<T: bytemuck::Pod> {
    /// Owned by cudarc (will cudaFree on drop).
    Owned(CudaSlice<T>),
    /// Pooled — raw device pointer + pool return channel.
    Pooled {
        size_bytes: usize,
        pool_id: u64,
        _marker: PhantomData<T>,
    },
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Wrap an owned `CudaSlice` from cudarc.
    /// The stream is needed to extract the device pointer at creation time.
    pub fn from_cuda_slice(slice: CudaSlice<T>, len: usize, stream: &Arc<CudaStream>) -> Self {
        let raw_ptr = {
            let (dev_ptr, _sync) = DevicePtr::device_ptr(&slice, stream);
            dev_ptr as u64
        };
        Self {
            inner: GpuBufferInner::Owned(slice),
            raw_ptr,
            len,
        }
    }

    /// Create from a raw pooled allocation.
    ///
    /// # Safety
    /// `ptr` must be a valid device pointer to at least `len * size_of::<T>()` bytes.
    pub unsafe fn from_pool(ptr: u64, len: usize, pool_id: u64) -> Self {
        Self {
            inner: GpuBufferInner::Pooled {
                size_bytes: len * std::mem::size_of::<T>(),
                pool_id,
                _marker: PhantomData,
            },
            raw_ptr: ptr,
            len,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Raw device pointer as u64.
    pub fn device_ptr(&self) -> u64 {
        self.raw_ptr
    }

    /// Raw device pointer as `*const T` for kernel launches.
    pub fn ptr(&self) -> *const T {
        self.raw_ptr as *const T
    }

    /// Raw mutable device pointer as `*mut T` for kernel launches.
    pub fn ptr_mut(&mut self) -> *mut T {
        self.raw_ptr as *mut T
    }

    /// Pool ID if this buffer came from a pool, else `None`.
    pub fn pool_id(&self) -> Option<u64> {
        match &self.inner {
            GpuBufferInner::Owned(_) => None,
            GpuBufferInner::Pooled { pool_id, .. } => Some(*pool_id),
        }
    }

    /// Convert to a `GpuBufferHandle` for cross-crate pipeline passing.
    pub fn to_handle(&self) -> nordocr_core::GpuBufferHandle {
        nordocr_core::GpuBufferHandle {
            ptr: self.raw_ptr as usize,
            size: self.size_bytes(),
            pool_id: self.pool_id().unwrap_or(0),
        }
    }
}

impl<T: bytemuck::Pod> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.len)
            .field("size_bytes", &self.size_bytes())
            .field("device_ptr", &format_args!("0x{:x}", self.raw_ptr))
            .finish()
    }
}
