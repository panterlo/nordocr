use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::CudaDevice;
use parking_lot::Mutex;
use std::sync::Arc;

use nordocr_core::{OcrError, Result};

use crate::buffer::GpuBuffer;

static POOL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Pre-allocated GPU memory pool using a slab allocator.
///
/// Avoids `cudaMalloc`/`cudaFree` during inference, as those calls
/// implicitly synchronize the device. Instead, we pre-allocate large
/// slabs at startup and carve out buffers from them.
pub struct GpuMemoryPool {
    id: u64,
    device: Arc<CudaDevice>,
    slabs: Mutex<Vec<Slab>>,
    free_lists: Mutex<BTreeMap<usize, Vec<u64>>>,
    slab_size: usize,
    total_allocated: AtomicU64,
}

struct Slab {
    /// Base device pointer for this slab.
    base_ptr: u64,
    /// Total size of the slab in bytes.
    size: usize,
    /// Current allocation offset within the slab (bump allocator).
    offset: usize,
}

/// Alignment for GPU allocations (256 bytes covers all TensorRT/CUDA requirements).
const GPU_ALLOC_ALIGN: usize = 256;

fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool with a given initial slab size.
    ///
    /// `initial_slab_size` is the size of each pre-allocated chunk (e.g. 256 MB).
    pub fn new(device: Arc<CudaDevice>, initial_slab_size: usize) -> Result<Self> {
        let id = POOL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut pool = Self {
            id,
            device,
            slabs: Mutex::new(Vec::new()),
            free_lists: Mutex::new(BTreeMap::new()),
            slab_size: initial_slab_size,
            total_allocated: AtomicU64::new(0),
        };

        // Allocate the first slab.
        pool.allocate_slab(initial_slab_size)?;

        Ok(pool)
    }

    /// Pool identity.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Total bytes allocated across all slabs.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Allocate a typed GPU buffer of `count` elements.
    pub fn alloc<T: bytemuck::Pod>(&self, count: usize) -> Result<GpuBuffer<T>> {
        let size = align_up(count * std::mem::size_of::<T>(), GPU_ALLOC_ALIGN);

        // Try free list first (exact or next-larger bucket).
        {
            let mut free = self.free_lists.lock();
            // Find the smallest bucket >= requested size.
            if let Some((&bucket_size, ptrs)) = free.range_mut(size..).next() {
                if let Some(ptr) = ptrs.pop() {
                    if ptrs.is_empty() {
                        let bucket_size_copy = bucket_size;
                        drop(ptrs);
                        free.remove(&bucket_size_copy);
                    }
                    return Ok(unsafe { GpuBuffer::from_pool(ptr, count, self.id) });
                }
            }
        }

        // Try bump-allocating from existing slabs.
        {
            let mut slabs = self.slabs.lock();
            for slab in slabs.iter_mut() {
                if let Some(ptr) = slab.try_alloc(size) {
                    return Ok(unsafe { GpuBuffer::from_pool(ptr, count, self.id) });
                }
            }
        }

        // All slabs exhausted — allocate a new slab.
        let new_slab_size = self.slab_size.max(size * 2);
        self.allocate_slab(new_slab_size)?;

        // Retry from the fresh slab.
        let mut slabs = self.slabs.lock();
        let slab = slabs.last_mut().expect("just allocated");
        let ptr = slab.try_alloc(size).ok_or(OcrError::GpuOutOfMemory {
            requested: size,
            available: 0,
        })?;

        Ok(unsafe { GpuBuffer::from_pool(ptr, count, self.id) })
    }

    /// Return a buffer to the pool's free list.
    pub fn free<T: bytemuck::Pod>(&self, buf: GpuBuffer<T>) {
        let size = align_up(buf.size_bytes(), GPU_ALLOC_ALIGN);
        let ptr = buf.device_ptr();

        let mut free = self.free_lists.lock();
        free.entry(size).or_default().push(ptr);
        // `buf` is dropped here but we do NOT call cudaFree — the pool owns the memory.
        std::mem::forget(buf);
    }

    fn allocate_slab(&self, size: usize) -> Result<()> {
        // Use cudarc to allocate raw device memory.
        let slice = self
            .device
            .alloc_zeros::<u8>(size)
            .map_err(|e| OcrError::Cuda(format!("slab alloc failed: {e}")))?;

        let base_ptr = *cudarc::driver::DevicePtr::device_ptr(&slice) as u64;

        // Prevent cudarc from freeing this — pool owns it.
        std::mem::forget(slice);

        self.slabs.lock().push(Slab {
            base_ptr,
            size,
            offset: 0,
        });

        self.total_allocated
            .fetch_add(size as u64, Ordering::Relaxed);

        tracing::debug!(
            pool_id = self.id,
            slab_size = size,
            total = self.total_allocated.load(Ordering::Relaxed),
            "allocated new GPU slab"
        );

        Ok(())
    }
}

impl Slab {
    fn try_alloc(&mut self, size: usize) -> Option<u64> {
        let aligned_offset = align_up(self.offset, GPU_ALLOC_ALIGN);
        if aligned_offset + size > self.size {
            return None;
        }
        let ptr = self.base_ptr + aligned_offset as u64;
        self.offset = aligned_offset + size;
        Some(ptr)
    }
}
