//! Conservative ACPI WDAT watchdog support for validated System-I/O registers.
//!
//! No chipset TCO base is guessed. Unsupported address spaces or action sequences
//! remain disarmed and use the safe no-op fallback.

use crate::smp;
use uefi::table::SystemTable;

const WDAT_FIXED_BYTES: usize = 68;
const WDAT_ENTRY_BYTES: usize = 24;
const WDAT_TARGET_DEADLINE_MS: u32 = 3_000;
const WDAT_ACTION_SET_COUNTDOWN: u8 = 4;
const WDAT_ACTION_SET_RUNNING: u8 = 6;
const WDAT_ACTION_SET_STOPPED: u8 = 8;
const WDAT_INSTRUCTION_WRITE_VALUE: u8 = 2;
const WDAT_INSTRUCTION_WRITE_COUNT: u8 = 3;
const SYSTEM_IO_ADDRESS_SPACE: u8 = 1;

#[derive(Clone, Copy)]
struct PortRegister {
    port: u16,
    width: u8,
    mask: u32,
    value: u32,
    preserve: bool,
}

#[derive(Clone, Copy)]
pub struct Watchdog {
    countdown: Option<PortRegister>,
    running: Option<PortRegister>,
    stopped: Option<PortRegister>,
    count: u32,
    armed: bool,
}

impl Watchdog {
    const fn unsupported() -> Self {
        Self {
            countdown: None,
            running: None,
            stopped: None,
            count: 0,
            armed: false,
        }
    }

    pub fn is_supported(&self) -> bool {
        self.countdown.is_some() && self.running.is_some() && self.stopped.is_some()
    }

    /// Arms the validated watchdog by loading its deadline and enabling it.
    pub fn arm(&mut self) -> bool {
        let (Some(countdown), Some(running), Some(_)) =
            (self.countdown, self.running, self.stopped)
        else {
            return false;
        };
        if !write_countdown(countdown, self.count) || !write_control(running) {
            crate::serial_println!(
                "[WATCHDOG WARNING]: WDAT arm failed; watchdog remains disarmed."
            );
            return false;
        }
        self.armed = true;
        crate::serial_println!(
            "[WATCHDOG]: WDAT armed; deadline={} ms, timer_count={}",
            WDAT_TARGET_DEADLINE_MS,
            self.count
        );
        true
    }

    /// Refreshes the deadline through the WDAT Set Countdown action.
    #[inline(always)]
    pub fn kick_watchdog(&self) {
        if self.armed {
            if let Some(register) = self.countdown {
                let _ = write_countdown(register, self.count);
            }
        }
    }

    /// Stops the watchdog when a finite appliance/QEMU run completes.
    pub fn disarm(&mut self) {
        if self.armed {
            if let Some(register) = self.stopped {
                let _ = write_control(register);
            }
            self.armed = false;
            crate::serial_println!("[WATCHDOG]: WDAT stopped after clean stream completion.");
        }
    }
}

/// Probe ACPI WDAT and accept only a complete, safe System-I/O action sequence.
pub fn probe(system_table: &SystemTable<uefi::table::Boot>) -> Watchdog {
    let Some((table, length)) = smp::find_acpi_table(system_table, b"WDAT") else {
        crate::serial_println!(
            "[WATCHDOG]: WDAT unavailable; safe no-op fallback (no chipset TCO base guessed)."
        );
        return Watchdog::unsupported();
    };
    if length < WDAT_FIXED_BYTES {
        crate::serial_println!("[WATCHDOG]: WDAT table truncated; safe no-op fallback.");
        return Watchdog::unsupported();
    }

    let timer_period_ms = read_u32(table + 48);
    let max_count = read_u32(table + 52);
    let min_count = read_u32(table + 56);
    let entry_count = read_u32(table + 64) as usize;
    let entries_bytes = match entry_count.checked_mul(WDAT_ENTRY_BYTES) {
        Some(bytes) => bytes,
        None => {
            crate::serial_println!("[WATCHDOG]: WDAT entry count overflow; safe no-op fallback.");
            return Watchdog::unsupported();
        }
    };
    if timer_period_ms == 0
        || timer_period_ms > WDAT_TARGET_DEADLINE_MS
        || min_count > max_count
        || entry_count == 0
        || entries_bytes > length - WDAT_FIXED_BYTES
    {
        crate::serial_println!("[WATCHDOG]: WDAT parameters unsupported; safe no-op fallback.");
        return Watchdog::unsupported();
    }

    let requested_count = WDAT_TARGET_DEADLINE_MS.div_ceil(timer_period_ms);
    if requested_count < min_count || requested_count > max_count {
        crate::serial_println!(
            "[WATCHDOG]: WDAT cannot represent a 3-second deadline; safe no-op fallback."
        );
        return Watchdog::unsupported();
    }

    let mut watchdog = Watchdog::unsupported();
    let mut index = 0usize;
    while index < entry_count {
        let entry = table + WDAT_FIXED_BYTES + index * WDAT_ENTRY_BYTES;
        let action = read_u8(entry);
        let instruction = read_u8(entry + 1);
        let preserve = instruction & 0x80 != 0;
        let instruction = instruction & 0x7F;
        let register = parse_port_register(
            entry + 4,
            read_u32(entry + 20),
            read_u32(entry + 16),
            preserve,
        );

        if let Some(register) = register {
            match (action, instruction) {
                (WDAT_ACTION_SET_COUNTDOWN, WDAT_INSTRUCTION_WRITE_COUNT) => {
                    watchdog.countdown = Some(register);
                }
                (WDAT_ACTION_SET_RUNNING, WDAT_INSTRUCTION_WRITE_VALUE)
                | (WDAT_ACTION_SET_RUNNING, WDAT_INSTRUCTION_WRITE_COUNT) => {
                    watchdog.running = Some(register);
                }
                (WDAT_ACTION_SET_STOPPED, WDAT_INSTRUCTION_WRITE_VALUE)
                | (WDAT_ACTION_SET_STOPPED, WDAT_INSTRUCTION_WRITE_COUNT) => {
                    watchdog.stopped = Some(register);
                }
                _ => {}
            }
        }
        index += 1;
    }

    watchdog.count = requested_count;
    if !watchdog.is_supported() {
        crate::serial_println!(
            "[WATCHDOG]: WDAT found but lacks validated System-I/O countdown/run/stop actions; safe no-op fallback."
        );
        return Watchdog::unsupported();
    }
    crate::serial_println!(
        "[WATCHDOG]: WDAT supported; System-I/O actions validated, period={} ms, deadline={} ms; awaiting arm.",
        timer_period_ms,
        WDAT_TARGET_DEADLINE_MS
    );
    watchdog
}

fn parse_port_register(gas: usize, mask: u32, value: u32, preserve: bool) -> Option<PortRegister> {
    let address_space = read_u8(gas);
    let width = read_u8(gas + 1);
    let bit_offset = read_u8(gas + 2);
    let access_size = read_u8(gas + 3);
    let address = read_u64(gas + 4);
    if address_space != SYSTEM_IO_ADDRESS_SPACE
        || bit_offset != 0
        || mask == 0
        || address > (u16::MAX - 3) as u64
        || !matches!(width, 8 | 16 | 32)
        || !matches!(access_size, 0 | 1 | 2 | 3)
    {
        return None;
    }
    let expected_access_size = match width {
        8 => 1,
        16 => 2,
        _ => 3,
    };
    if access_size != 0 && access_size != expected_access_size {
        return None;
    }
    Some(PortRegister {
        port: address as u16,
        width,
        mask,
        value,
        preserve,
    })
}

fn write_countdown(register: PortRegister, count: u32) -> bool {
    let shift = register.mask.trailing_zeros();
    let encoded = count.checked_shl(shift).unwrap_or(0) & register.mask;
    write_masked(register, encoded)
}

fn write_control(register: PortRegister) -> bool {
    write_masked(register, register.value & register.mask)
}

fn write_masked(register: PortRegister, value: u32) -> bool {
    let old = if register.preserve {
        match read_port(register) {
            Some(old) => old,
            None => return false,
        }
    } else {
        0
    };
    let value = (old & !register.mask) | (value & register.mask);
    write_port(register, value)
}

fn read_port(register: PortRegister) -> Option<u32> {
    let value: u32;
    unsafe {
        match register.width {
            8 => {
                let byte: u8;
                core::arch::asm!("in al, dx", in("dx") register.port, out("al") byte, options(nomem, nostack, preserves_flags));
                value = byte as u32;
            }
            16 => {
                let word: u16;
                core::arch::asm!("in ax, dx", in("dx") register.port, out("ax") word, options(nomem, nostack, preserves_flags));
                value = word as u32;
            }
            32 => {
                core::arch::asm!("in eax, dx", in("dx") register.port, out("eax") value, options(nomem, nostack, preserves_flags));
            }
            _ => return None,
        }
    }
    Some(value)
}

fn write_port(register: PortRegister, value: u32) -> bool {
    unsafe {
        match register.width {
            8 => {
                core::arch::asm!("out dx, al", in("dx") register.port, in("al") value as u8, options(nomem, nostack, preserves_flags))
            }
            16 => {
                core::arch::asm!("out dx, ax", in("dx") register.port, in("ax") value as u16, options(nomem, nostack, preserves_flags))
            }
            32 => {
                core::arch::asm!("out dx, eax", in("dx") register.port, in("eax") value, options(nomem, nostack, preserves_flags))
            }
            _ => return false,
        }
    }
    true
}

fn read_u8(address: usize) -> u8 {
    unsafe { core::ptr::read_volatile(address as *const u8) }
}

fn read_u32(address: usize) -> u32 {
    unsafe { core::ptr::read_unaligned(address as *const u32) }
}

fn read_u64(address: usize) -> u64 {
    unsafe { core::ptr::read_unaligned(address as *const u64) }
}
