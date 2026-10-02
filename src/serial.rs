//! Direct Hardware Serial Telemetry (16550 UART driver for COM1).
//!
//! Provides raw, zero-dependency serial communication over I/O port 0x3F8,
//! allowing telemetry even after UEFI boot services are terminated.

use core::fmt;

/// Standard COM1 base I/O port.
pub const COM1_BASE: u16 = 0x3F8;
pub const COM2_BASE: u16 = 0x2F8;
pub const INPUT_PREAMBLE: [u8; 2] = *b"NB";
pub const OUTPUT_PREAMBLE: [u8; 2] = *b"NR";

/// Line Status Register bit 5: Transmitter Holding Register Empty (THRE).
pub const LSR_THRE: u8 = 0x20;

/// Inline assembly helper to write a byte to an x86 I/O port.
#[inline(always)]
pub unsafe fn outb(port: u16, val: u8) {
    core::arch::asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack, preserves_flags)
    );
}

/// Inline assembly helper to read a byte from an x86 I/O port.
#[inline(always)]
pub unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    core::arch::asm!(
        "in al, dx",
        in("dx") port,
        out("al") val,
        options(nomem, nostack, preserves_flags)
    );
    val
}

/// Inline assembly helper to write a 16-bit word to an x86 I/O port.
#[inline(always)]
pub unsafe fn outw(port: u16, val: u16) {
    core::arch::asm!(
        "out dx, ax",
        in("dx") port,
        in("ax") val,
        options(nomem, nostack, preserves_flags)
    );
}

/// Initializes standard COM1 UART (0x3F8) for 115200 baud, 8 data bits, no parity, 1 stop bit (8-N-1).
/// Interrupts are disabled and the 14-byte FIFO buffer is enabled.
pub unsafe fn init() {
    init_port(COM1_BASE, false);
}

/// Initializes a second UART as the binary streaming port.
pub unsafe fn init_wire_port() {
    init_port(COM2_BASE, false);
}

/// Initializes a 16550-compatible UART at the supplied base I/O port.
pub unsafe fn init_port(base: u16, enable_rx_interrupt: bool) {
    // 1. Disable all UART interrupts (IER = 0x00)
    outb(base + 1, 0x00);

    // 2. Enable Divisor Latch Access Bit (DLAB = 1) in Line Control Register
    outb(base + 3, 0x80);

    // 3. Set baud rate divisor to 1 (115200 baud = 115200 / 1)
    outb(base, 0x01); // Divisor latch low byte
    outb(base + 1, 0x00); // Divisor latch high byte

    // 4. Configure 8 bits, no parity, 1 stop bit (8-N-1) and clear DLAB (0x03)
    outb(base + 3, 0x03);

    // 5. Enable FIFO, clear TX/RX FIFO queues, set 14-byte threshold (0xC7)
    outb(base + 2, 0xC7);

    // 6. Set RTS and DTR, enable Auxiliary Output 2 (0x0B)
    outb(base + 4, 0x0B);

    if enable_rx_interrupt {
        outb(base + 1, 0x01);
    }
}

/// Enables Receiver Data Available in IER. Polling remains the consumer; callers
/// should keep CPU interrupts disabled unless an IRQ handler has been installed.
pub unsafe fn enable_rx_interrupt(base: u16) {
    outb(base + 1, 0x01);
}

/// Returns a COM1 byte if the UART reports Data Ready.
#[inline(always)]
pub fn poll_byte() -> Option<u8> {
    poll_byte_from(COM1_BASE)
}

/// Returns one byte without blocking when the selected UART reports Data Ready.
#[inline(always)]
pub fn poll_byte_from(base: u16) -> Option<u8> {
    unsafe {
        if inb(base + 5) & 0x01 != 0 {
            Some(inb(base))
        } else {
            None
        }
    }
}

/// Fixed-length frame reader for `NB` followed by 64 raw signed-byte values.
pub struct FrameReader {
    state: u8,
    index: usize,
    payload: [i8; 64],
    preamble_tsc: u64,
}

pub struct ReceivedFrame {
    pub payload: [i8; 64],
    pub preamble_tsc: u64,
}

pub enum FrameEvent {
    Input(ReceivedFrame),
    Reset { preamble_tsc: u64 },
}

impl FrameReader {
    pub const fn new() -> Self {
        Self {
            state: 0,
            index: 0,
            payload: [0; 64],
            preamble_tsc: 0,
        }
    }

    /// Consumes available UART bytes and returns a complete frame when found.
    pub fn poll_frame(&mut self, base: u16) -> Option<FrameEvent> {
        loop {
            let byte = poll_byte_from(base)?;
            if let Some(event) = self.consume_byte(byte) {
                return Some(event);
            }
        }
    }

    fn consume_byte(&mut self, byte: u8) -> Option<FrameEvent> {
        match self.state {
            0 => {
                if byte == INPUT_PREAMBLE[0] {
                    self.state = 1;
                    self.preamble_tsc = unsafe { crate::timer::read_tsc() };
                }
            }
            1 => {
                if byte == INPUT_PREAMBLE[1] {
                    self.state = 2;
                    self.index = 0;
                } else if byte == OUTPUT_PREAMBLE[1] {
                    self.state = 0;
                    return Some(FrameEvent::Reset {
                        preamble_tsc: self.preamble_tsc,
                    });
                } else if byte == INPUT_PREAMBLE[0] {
                    self.preamble_tsc = unsafe { crate::timer::read_tsc() };
                } else {
                    self.state = 0;
                }
            }
            _ => {
                self.payload[self.index] = byte as i8;
                self.index += 1;
                if self.index == self.payload.len() {
                    self.state = 0;
                    self.index = 0;
                    return Some(FrameEvent::Input(ReceivedFrame {
                        payload: self.payload,
                        preamble_tsc: self.preamble_tsc,
                    }));
                }
            }
        }
        None
    }
}

/// Writes a single byte to COM1, translating `\n` to `\r\n` for standard serial terminals.
/// Spins until the Transmitter Holding Register Empty (THRE) bit is set.
pub unsafe fn write_byte(b: u8) {
    if b == b'\n' {
        write_byte_raw_from(COM1_BASE, b'\r');
    }
    write_byte_raw_from(COM1_BASE, b);
}

/// Writes a raw byte to a UART without newline translation.
#[inline(always)]
pub unsafe fn write_byte_raw_from(base: u16, b: u8) {
    // Wait until LSR bit 5 (Transmitter Holding Register Empty) is 1
    while (inb(base + 5) & LSR_THRE) == 0 {
        core::hint::spin_loop();
    }
    outb(base, b);
}

/// Writes binary bytes without text newline translation.
pub unsafe fn write_raw_bytes_from(base: u16, bytes: &[u8]) {
    for &byte in bytes {
        write_byte_raw_from(base, byte);
    }
}

/// Writes a string slice to COM1.
pub unsafe fn write_str(s: &str) {
    for &b in s.as_bytes() {
        write_byte(b);
    }
}

/// Zero-sized type implementing `core::fmt::Write` for formatted serial printing.
pub struct SerialPort;

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        unsafe {
            crate::serial::write_str(s);
        }
        Ok(())
    }
}

/// Formatted print macro for COM1 serial telemetry.
#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut serial = $crate::serial::SerialPort;
            let _ = write!(serial, $($arg)*);
        }
    };
}

/// Formatted println macro for COM1 serial telemetry with newline.
#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($($arg:tt)*) => {
        $crate::serial_print!("{}\n", format_args!($($arg)*))
    };
}
