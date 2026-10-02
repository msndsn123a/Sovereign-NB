//! Fixed-capacity, allocation-free single-producer/single-consumer queues.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

pub const RING_CAPACITY: usize = 128;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputFrame {
    pub payload: [i8; 64],
    pub t0_preamble: u64,
    pub t1_ingress: u64,
    pub stream_id: u8,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputFrame {
    pub output_dim: u8,
    pub values: [i32; 64],
    pub t0_preamble: u64,
    pub t1_ingress: u64,
    pub t2_compute: u64,
}

impl OutputFrame {
    pub const ZERO: Self = Self {
        output_dim: 0,
        values: [0; 64],
        t0_preamble: 0,
        t1_ingress: 0,
        t2_compute: 0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingFull;

/// A bounded SPSC ring. Split it once while exclusively borrowed, then move
/// the producer and consumer handles to their respective execution contexts.
pub struct SpscRing<T: Copy> {
    head: AtomicUsize,
    tail: AtomicUsize,
    slots: [UnsafeCell<MaybeUninit<T>>; RING_CAPACITY],
}

unsafe impl<T: Copy + Send> Sync for SpscRing<T> {}

impl<T: Copy> SpscRing<T> {
    pub const fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            slots: [const { UnsafeCell::new(MaybeUninit::uninit()) }; RING_CAPACITY],
        }
    }

    pub fn split(&mut self) -> (Producer<'_, T>, Consumer<'_, T>) {
        (Producer { ring: self }, Consumer { ring: self })
    }

    /// Snapshot queue cursors through a raw pointer without creating an aliasing reference.
    ///
    /// # Safety
    /// `ring` must point to a live `SpscRing<T>` for the duration of the atomic reads.
    pub unsafe fn snapshot_raw(ring: *const Self) -> (usize, usize) {
        let head = (*ring).head.load(Ordering::Acquire);
        let tail = (*ring).tail.load(Ordering::Acquire);
        (head, tail)
    }
}

pub struct Producer<'a, T: Copy> {
    ring: &'a SpscRing<T>,
}

impl<T: Copy> Producer<'_, T> {
    pub fn push(&mut self, value: T) -> Result<(), RingFull> {
        self.push_with(value, |_| {})
    }

    /// Writes a value, lets the producer stamp it while the slot is private,
    /// then publishes the slot with Release ordering.
    pub fn push_with<F>(&mut self, value: T, stamp: F) -> Result<(), RingFull>
    where
        F: FnOnce(&mut T),
    {
        let head = self.ring.head.load(Ordering::Relaxed);
        let tail = self.ring.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= RING_CAPACITY {
            return Err(RingFull);
        }

        let slot = head % RING_CAPACITY;
        unsafe {
            let storage = &mut *self.ring.slots[slot].get();
            storage.write(value);
            stamp(storage.assume_init_mut());
        }
        self.ring
            .head
            .store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

pub struct Consumer<'a, T: Copy> {
    ring: &'a SpscRing<T>,
}

impl<T: Copy> Consumer<'_, T> {
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.ring.tail.load(Ordering::Relaxed);
        let head = self.ring.head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }

        let slot = tail % RING_CAPACITY;
        let value = unsafe { (*self.ring.slots[slot].get()).assume_init_read() };
        self.ring
            .tail
            .store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }
}

pub type InputRing = SpscRing<InputFrame>;
pub type OutputRing = SpscRing<OutputFrame>;
