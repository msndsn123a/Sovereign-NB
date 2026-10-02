//! CPUID-gated UMONITOR/UMWAIT support with a bounded PAUSE fallback.

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

const CPUID_WAITPKG: u32 = 1 << 5;
const IA32_UMWAIT_CONTROL: u32 = 0xE1;
const UMWAIT_C02_DISABLE: u64 = 1;
const MAX_WAIT_CYCLES: u64 = 2048;
const PAUSE_BATCH: usize = 64;

static WAITPKG_ACTIVE: AtomicBool = AtomicBool::new(false);
static WAITPKG_MONITOR: AtomicU32 = AtomicU32::new(0);
static UMWAIT_CONTROL_VALUE: AtomicU32 = AtomicU32::new(0);

#[inline]
pub fn cpu_supports_waitpkg() -> bool {
    if core::arch::x86_64::__cpuid(0).eax < 7 {
        return false;
    }
    core::arch::x86_64::__cpuid_count(7, 0).ecx & CPUID_WAITPKG != 0
}

unsafe fn read_msr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    core::arch::asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | low as u64
}

unsafe fn write_msr(msr: u32, value: u64) {
    core::arch::asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") value as u32,
        in("edx") (value >> 32) as u32,
        options(nomem, nostack, preserves_flags),
    );
}

/// Initialize WAITPKG only after its CPUID feature bit is present.
///
/// The MSR's C0.2-disable bit is cleared when set, then validated by readback.
/// The max-time field is preserved; each instruction also has an explicit short
/// TSC deadline so a missed monitor wake cannot stall ingress indefinitely.
pub fn initialize() -> bool {
    if !cpu_supports_waitpkg() {
        WAITPKG_ACTIVE.store(false, Ordering::Release);
        return false;
    }

    let mut control = unsafe { read_msr(IA32_UMWAIT_CONTROL) };
    if control & UMWAIT_C02_DISABLE != 0 {
        unsafe { write_msr(IA32_UMWAIT_CONTROL, control & !UMWAIT_C02_DISABLE) };
        control = unsafe { read_msr(IA32_UMWAIT_CONTROL) };
    }
    UMWAIT_CONTROL_VALUE.store(control as u32, Ordering::Relaxed);
    let active = control & UMWAIT_C02_DISABLE == 0;
    WAITPKG_ACTIVE.store(active, Ordering::Release);
    active
}

pub fn is_active() -> bool {
    WAITPKG_ACTIVE.load(Ordering::Acquire)
}

pub fn control_value() -> u32 {
    UMWAIT_CONTROL_VALUE.load(Ordering::Relaxed)
}

/// Wait for a store to `address`, using C0.2 and a bounded TSC deadline.
///
/// # Safety
/// `address` must be a valid, naturally aligned, cacheable monitor address for
/// the current privilege context. Do not pass an uncacheable PCI MMIO register.
pub unsafe fn wait_for_address(address: *const u8) {
    if !is_active() {
        bounded_pause();
        return;
    }
    waitpkg_wait(address);
}

/// A bounded low-power wait for transports without a monitorable producer word.
pub fn bounded_wait() {
    if !is_active() {
        bounded_pause();
        return;
    }
    let address = core::ptr::addr_of!(WAITPKG_MONITOR).cast::<u8>();
    unsafe { waitpkg_wait(address) };
}

#[inline(always)]
fn bounded_pause() {
    let mut count = 0usize;
    while count < PAUSE_BATCH {
        core::hint::spin_loop();
        count += 1;
    }
}

#[inline(always)]
unsafe fn waitpkg_wait(address: *const u8) {
    let deadline = crate::timer::read_tsc().wrapping_add(MAX_WAIT_CYCLES);
    core::arch::asm!(
        "umonitor rax",
        in("rax") address as u64,
        options(nostack),
    );
    core::sync::atomic::fence(Ordering::SeqCst);
    // ECX=0 requests the deepest supported C0 substate (C0.2).
    core::arch::asm!(
        "umwait ecx",
        in("ecx") 0u32,
        in("eax") deadline as u32,
        in("edx") (deadline >> 32) as u32,
        options(nostack),
    );
}
