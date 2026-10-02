//! Minimal AArch64 UEFI platform entry. Firmware-owned translation tables are retained.

use core::fmt::Write;
use uefi::prelude::*;

#[inline(always)]
pub unsafe fn read_counter() -> u64 {
    let value: u64;
    core::arch::asm!(
        "isb",
        "mrs {value}, cntvct_el0",
        value = out(reg) value,
        options(nomem, nostack, preserves_flags),
    );
    value
}

#[inline(always)]
pub unsafe fn counter_frequency_hz() -> u64 {
    let value: u64;
    core::arch::asm!(
        "mrs {value}, cntfrq_el0",
        value = out(reg) value,
        options(nomem, nostack, preserves_flags),
    );
    value
}

pub fn run(_image_handle: Handle, mut system_table: SystemTable<Boot>) -> Status {
    let counter_hz = unsafe { counter_frequency_hz() };
    let counter = unsafe { read_counter() };
    let neon_parity = crate::neon_kernel::parity_self_test();
    {
        let stdout = system_table.stdout();
        let _ = writeln!(stdout, "SOVEREIGN-NB AArch64 UEFI: firmware entry active");
        let _ = writeln!(
            stdout,
            "[ARM TIMER]: CNTVCT_EL0={}, CNTFRQ_EL0={} Hz",
            counter, counter_hz
        );
        let _ = writeln!(
            stdout,
            "[ARM NEON VERIFY]: scalar/NEON exact parity={}, vectors=3, activation_modes=2",
            neon_parity
        );
    }

    if !neon_parity || counter_hz == 0 {
        return Status::ABORTED;
    }

    // This wrapper captures a fresh map and retries ExitBootServices on a stale key.
    // Keep the returned map alive; after this point no UEFI protocol is used and
    // the firmware's AArch64 page tables remain untouched.
    let _memory_map =
        unsafe { uefi::boot::exit_boot_services(uefi::table::boot::MemoryType::LOADER_DATA) };
    loop {
        let _ = unsafe { read_counter() };
        core::hint::spin_loop();
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
