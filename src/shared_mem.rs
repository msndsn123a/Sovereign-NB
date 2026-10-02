//! Cacheline-aligned shared mailbox for a single external producer and consumer.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const MAILBOX_MAGIC: u32 = 0x5348_4D42;
pub const MAILBOX_VERSION: u32 = 1;
pub const MAILBOX_CAPACITY: usize = 16;
pub const HOST_MAILBOX_PHYSICAL_BASE: u64 = 0x1_0000_0000;
pub const HOST_MAILBOX_WINDOW_SIZE: usize = 64 * 1024;
pub const INPUT_EMPTY: u32 = 0;
pub const INPUT_READY: u32 = 1;
pub const OUTPUT_EMPTY: u32 = 0;
pub const OUTPUT_READY: u32 = 1;

#[repr(C, align(64))]
pub struct InputSlot {
    pub state: AtomicU32,
    pub _reserved: u32,
    pub t0_ready: AtomicU64,
    pub _cacheline_padding: [u8; 48],
    pub payload: UnsafeCell<[i8; 64]>,
}

impl InputSlot {
    pub const fn new() -> Self {
        Self {
            state: AtomicU32::new(INPUT_EMPTY),
            _reserved: 0,
            t0_ready: AtomicU64::new(0),
            _cacheline_padding: [0; 48],
            payload: UnsafeCell::new([0; 64]),
        }
    }

    /// The caller must have observed INPUT_READY with Acquire and must not
    /// release the slot back to its producer until it has finished reading.
    pub unsafe fn payload(&self) -> &[i8; 64] {
        &*self.payload.get()
    }
}

unsafe impl Sync for InputSlot {}

#[repr(C, align(64))]
pub struct OutputSlot {
    pub state: AtomicU32,
    pub _state_padding: [u8; 60],
    pub values: UnsafeCell<[i32; 64]>,
    pub metadata: UnsafeCell<OutputMetadata>,
    pub t3_commit: AtomicU64,
    pub _tail_padding: [u8; 24],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct OutputMetadata {
    pub output_dim: u32,
    pub t0_ready: u64,
    pub t1_ingest: u64,
    pub t2_compute: u64,
}

impl OutputSlot {
    pub const fn new() -> Self {
        Self {
            state: AtomicU32::new(OUTPUT_EMPTY),
            _state_padding: [0; 60],
            values: UnsafeCell::new([0; 64]),
            metadata: UnsafeCell::new(OutputMetadata {
                output_dim: 0,
                t0_ready: 0,
                t1_ingest: 0,
                t2_compute: 0,
            }),
            t3_commit: AtomicU64::new(0),
            _tail_padding: [0; 24],
        }
    }
}

unsafe impl Sync for OutputSlot {}

#[repr(C, align(64))]
pub struct SharedMailbox {
    pub magic: u32,
    pub version: u32,
    pub input_head: AtomicU64,
    pub input_tail: AtomicU64,
    pub output_head: AtomicU64,
    pub output_tail: AtomicU64,
    pub _header_padding: [u8; 24],
    pub input_slots: [InputSlot; MAILBOX_CAPACITY],
    pub output_slots: [OutputSlot; MAILBOX_CAPACITY],
}

impl SharedMailbox {
    pub const fn new() -> Self {
        Self {
            magic: MAILBOX_MAGIC,
            version: MAILBOX_VERSION,
            input_head: AtomicU64::new(0),
            input_tail: AtomicU64::new(0),
            output_head: AtomicU64::new(0),
            output_tail: AtomicU64::new(0),
            _header_padding: [0; 24],
            input_slots: [const { InputSlot::new() }; MAILBOX_CAPACITY],
            output_slots: [const { OutputSlot::new() }; MAILBOX_CAPACITY],
        }
    }

    /// Binds to a host-backed guest RAM window at the configured fixed GPA.
    ///
    /// # Safety
    /// The platform must map a writable, at least `size_of::<SharedMailbox>()`
    /// region at this physical address and guarantee cache-coherent shared RAM.
    pub unsafe fn bind_host_window(physical_address: u64) -> Result<core::ptr::NonNull<Self>, ()> {
        if physical_address % core::mem::align_of::<Self>() as u64 != 0
            || core::mem::size_of::<Self>() > HOST_MAILBOX_WINDOW_SIZE
        {
            return Err(());
        }
        let pointer = core::ptr::NonNull::new(physical_address as *mut Self).ok_or(())?;
        pointer.as_ptr().write(Self::new());
        Ok(pointer)
    }

    /// Shared-producer operation. Payload bytes are written into their final slot.
    pub fn try_publish_input(&self, payload: &[i8; 64], t0_ready: u64) -> Result<u64, ()> {
        let head = self.input_head.load(Ordering::Relaxed);
        let tail = self.input_tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= MAILBOX_CAPACITY as u64 {
            return Err(());
        }

        let slot = &self.input_slots[head as usize % MAILBOX_CAPACITY];
        if slot.state.load(Ordering::Acquire) != INPUT_EMPTY {
            return Err(());
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                (*slot.payload.get()).as_mut_ptr(),
                64,
            );
        }
        slot.t0_ready.store(t0_ready, Ordering::Relaxed);
        slot.state.store(INPUT_READY, Ordering::Release);
        self.input_head
            .store(head.wrapping_add(1), Ordering::Release);
        Ok(head)
    }

    /// Returns the next committed input slot without copying its payload.
    /// T0 is sampled immediately after the consumer observes READY with Acquire.
    pub fn peek_input(&self) -> Option<(u64, &InputSlot, u64)> {
        let tail = self.input_tail.load(Ordering::Relaxed);
        let head = self.input_head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let slot = &self.input_slots[tail as usize % MAILBOX_CAPACITY];
        if slot.state.load(Ordering::Acquire) != INPUT_READY {
            return None;
        }
        let t0_observed = unsafe { crate::timer::read_tsc() };
        Some((tail, slot, t0_observed))
    }

    pub fn consume_input(&self, sequence: u64) {
        let slot = &self.input_slots[sequence as usize % MAILBOX_CAPACITY];
        slot.state.store(INPUT_EMPTY, Ordering::Release);
        self.input_tail
            .store(sequence.wrapping_add(1), Ordering::Release);
    }

    /// Publishes a result directly to its output slot, then advances output_head.
    pub fn publish_output(
        &self,
        output_dim: usize,
        values: &[i32; 64],
        t0_ready: u64,
        t1_ingest: u64,
        t2_compute: u64,
        t3_commit: u64,
    ) -> Result<u64, ()> {
        let head = self.output_head.load(Ordering::Relaxed);
        let tail = self.output_tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= MAILBOX_CAPACITY as u64 || output_dim > 64 {
            return Err(());
        }

        let slot = &self.output_slots[head as usize % MAILBOX_CAPACITY];
        if slot.state.load(Ordering::Acquire) != OUTPUT_EMPTY {
            return Err(());
        }
        unsafe {
            core::ptr::copy_nonoverlapping(values.as_ptr(), (*slot.values.get()).as_mut_ptr(), 64);
        }
        unsafe {
            *slot.metadata.get() = OutputMetadata {
                output_dim: output_dim as u32,
                t0_ready,
                t1_ingest,
                t2_compute,
            };
        }
        slot.t3_commit.store(t3_commit, Ordering::Relaxed);
        slot.state.store(OUTPUT_READY, Ordering::Release);
        self.output_head
            .store(head.wrapping_add(1), Ordering::Release);
        Ok(head)
    }

    /// Computes directly into the next shared output slot and publishes it.
    /// The callback writes the final output array in-place and returns T2.
    pub fn compute_and_publish_output<F>(
        &self,
        output_dim: usize,
        t0_ready: u64,
        t1_ingest: u64,
        compute: F,
    ) -> Result<(u64, u64, u64), ()>
    where
        F: FnOnce(&mut [i32; 64]) -> u64,
    {
        let head = self.output_head.load(Ordering::Relaxed);
        let tail = self.output_tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= MAILBOX_CAPACITY as u64 || output_dim > 64 {
            return Err(());
        }

        let slot = &self.output_slots[head as usize % MAILBOX_CAPACITY];
        if slot.state.load(Ordering::Acquire) != OUTPUT_EMPTY {
            return Err(());
        }
        let t2_compute = unsafe { compute(&mut *slot.values.get()) };
        unsafe {
            *slot.metadata.get() = OutputMetadata {
                output_dim: output_dim as u32,
                t0_ready,
                t1_ingest,
                t2_compute,
            };
        }
        let t3_commit = unsafe { crate::timer::read_tsc() };
        slot.t3_commit.store(t3_commit, Ordering::Relaxed);
        slot.state.store(OUTPUT_READY, Ordering::Release);
        self.output_head
            .store(head.wrapping_add(1), Ordering::Release);
        Ok((head, t2_compute, t3_commit))
    }

    /// Test/consumer operation: copy a committed output then release its slot.
    pub fn consume_output(&self) -> Option<(u64, [i32; 64], u32)> {
        let tail = self.output_tail.load(Ordering::Relaxed);
        let head = self.output_head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let slot = &self.output_slots[tail as usize % MAILBOX_CAPACITY];
        if slot.state.load(Ordering::Acquire) != OUTPUT_READY {
            return None;
        }
        let mut values = [0i32; 64];
        unsafe {
            core::ptr::copy_nonoverlapping((*slot.values.get()).as_ptr(), values.as_mut_ptr(), 64);
        }
        let output_dim = unsafe { (*slot.metadata.get()).output_dim };
        slot.state.store(OUTPUT_EMPTY, Ordering::Release);
        self.output_tail
            .store(tail.wrapping_add(1), Ordering::Release);
        Some((tail, values, output_dim))
    }
}

const _: () = assert!(core::mem::align_of::<SharedMailbox>() == 64);
const _: () = assert!(core::mem::size_of::<InputSlot>() % 64 == 0);
const _: () = assert!(core::mem::size_of::<OutputSlot>() % 64 == 0);
const _: () = assert!(core::mem::offset_of!(InputSlot, payload) % 64 == 0);
const _: () = assert!(core::mem::offset_of!(OutputSlot, values) % 64 == 0);
const _: () = assert!(core::mem::offset_of!(SharedMailbox, input_slots) % 64 == 0);
const _: () = assert!(core::mem::offset_of!(SharedMailbox, output_slots) % 64 == 0);
