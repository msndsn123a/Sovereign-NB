//! Minimal PCI configuration space driver and NVMe device detection.
//!
//! Provides direct I/O port (0xCF8 / 0xCFC) access to enumerate PCI devices,
//! locate NVM Express storage controllers, and enable memory space + bus mastering.

/// PCI Configuration Mechanism I/O Ports
pub const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
pub const PCI_CONFIG_DATA: u16 = 0xCFC;

/// Information for an identified PCI device.
#[derive(Clone, Copy, Debug)]
pub struct PciDeviceInfo {
    pub bus: u8,
    pub slot: u8,
    pub func: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub bar0_phys: u64,
    pub bar0_mmio: *mut u8,
}

/// QEMU IVSHMEM device exposing a shared-memory aperture in PCI BAR2.
#[derive(Clone, Copy, Debug)]
pub struct PciSharedMemoryInfo {
    pub bus: u8,
    pub slot: u8,
    pub func: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub bar0_phys: u64,
    pub shared_memory_phys: u64,
    pub shared_memory_size: u64,
}

fn bar_base(bus: u8, slot: u8, func: u8, index: u8) -> Option<(u64, bool)> {
    let offset = 0x10 + index * 4;
    let low = unsafe { pci_read_u32(bus, slot, func, offset) };
    if low == 0 || low == u32::MAX || low & 1 != 0 {
        return None;
    }
    let is_64bit = low & 0x06 == 0x04;
    let high = if is_64bit {
        unsafe { pci_read_u32(bus, slot, func, offset + 4) }
    } else {
        0
    };
    Some((((high as u64) << 32) | (low as u64 & !0x0F), is_64bit))
}

/// Measures a memory BAR's size using the PCI sizing-mask procedure.
/// The previous command and BAR values are restored before returning.
unsafe fn bar_size(bus: u8, slot: u8, func: u8, index: u8, is_64bit: bool) -> Option<u64> {
    let offset = 0x10 + index * 4;
    let saved_command = pci_read_u32(bus, slot, func, 0x04);
    let saved_low = pci_read_u32(bus, slot, func, offset);
    let saved_high = if is_64bit {
        pci_read_u32(bus, slot, func, offset + 4)
    } else {
        0
    };

    // Disable memory decoding while probing to avoid exposing a temporary BAR value.
    pci_write_u32(bus, slot, func, 0x04, saved_command & !(1 << 1));
    pci_write_u32(bus, slot, func, offset, u32::MAX);
    if is_64bit {
        pci_write_u32(bus, slot, func, offset + 4, u32::MAX);
    }
    let mask_low = pci_read_u32(bus, slot, func, offset) & !0x0F;
    let mask_high = if is_64bit {
        pci_read_u32(bus, slot, func, offset + 4)
    } else {
        0
    };
    pci_write_u32(bus, slot, func, offset, saved_low);
    if is_64bit {
        pci_write_u32(bus, slot, func, offset + 4, saved_high);
    }
    pci_write_u32(bus, slot, func, 0x04, saved_command);

    let mask = ((mask_high as u64) << 32) | mask_low as u64;
    if mask == 0 {
        return None;
    }
    Some((!mask).wrapping_add(1))
}

/// Finds the QEMU IVSHMEM device (`1AF4:1110`) and returns BAR2, its shared-RAM aperture.
/// BAR0 is reported separately because it normally contains device registers, not the ring.
pub unsafe fn find_ivshmem_device() -> Option<PciSharedMemoryInfo> {
    for bus in 0..=u8::MAX {
        for slot in 0..32 {
            for func in 0..8 {
                let id = pci_read_u32(bus, slot, func, 0x00);
                let vendor_id = id as u16;
                if vendor_id == u16::MAX {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let device_id = (id >> 16) as u16;
                if vendor_id == 0x1AF4 && device_id == 0x1110 {
                    let (bar0_phys, _) = bar_base(bus, slot, func, 0)?;
                    let (shared_memory_phys, is_64bit) = bar_base(bus, slot, func, 2)?;
                    let shared_memory_size = bar_size(bus, slot, func, 2, is_64bit)?;
                    if shared_memory_phys == 0 || shared_memory_size == 0 {
                        return None;
                    }

                    let mut command = pci_read_u32(bus, slot, func, 0x04);
                    command |= (1 << 1) | (1 << 2);
                    pci_write_u32(bus, slot, func, 0x04, command);
                    return Some(PciSharedMemoryInfo {
                        bus,
                        slot,
                        func,
                        vendor_id,
                        device_id,
                        bar0_phys,
                        shared_memory_phys,
                        shared_memory_size,
                    });
                }

                if func == 0 {
                    let header_type = (pci_read_u32(bus, slot, 0, 0x0C) >> 16) & 0xFF;
                    if header_type & 0x80 == 0 {
                        break;
                    }
                }
            }
        }
    }
    None
}

/// Helper to write a 32-bit dword to an x86 I/O port.
#[inline(always)]
unsafe fn outl(port: u16, val: u32) {
    core::arch::asm!(
        "out dx, eax",
        in("dx") port,
        in("eax") val,
        options(nomem, nostack, preserves_flags)
    );
}

/// Helper to read a 32-bit dword from an x86 I/O port.
#[inline(always)]
unsafe fn inl(port: u16) -> u32 {
    let val: u32;
    core::arch::asm!(
        "in eax, dx",
        in("dx") port,
        out("eax") val,
        options(nomem, nostack, preserves_flags)
    );
    val
}

/// Reads a 32-bit double-word from PCI configuration space at the given offset.
pub unsafe fn pci_read_u32(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | ((slot as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);
    outl(PCI_CONFIG_ADDRESS, address);
    inl(PCI_CONFIG_DATA)
}

/// Writes a 32-bit double-word to PCI configuration space at the given offset.
pub unsafe fn pci_write_u32(bus: u8, slot: u8, func: u8, offset: u8, val: u32) {
    let address = (1u32 << 31)
        | ((bus as u32) << 16)
        | ((slot as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xFC);
    outl(PCI_CONFIG_ADDRESS, address);
    outl(PCI_CONFIG_DATA, val);
}

/// Iterates through PCI buses (0..8) and devices (0..32) to locate an NVMe controller:
/// - Class Code 0x01 (Mass Storage)
/// - Subclass 0x08 (Non-Volatile Memory)
/// - Programming Interface 0x02 (NVM Express)
///
/// Enables Bus Mastering and Memory Space access in the PCI Command register.
pub unsafe fn find_nvme_device() -> Option<PciDeviceInfo> {
    for bus in 0..8 {
        for slot in 0..32 {
            for func in 0..8 {
                let id_reg = pci_read_u32(bus, slot, func, 0x00);
                let vendor_id = (id_reg & 0xFFFF) as u16;
                if vendor_id == 0xFFFF {
                    if func == 0 {
                        break; // Slot is empty, skip remaining functions
                    }
                    continue;
                }

                let class_reg = pci_read_u32(bus, slot, func, 0x08);
                let class_code = (class_reg >> 24) as u8;
                let subclass = ((class_reg >> 16) & 0xFF) as u8;
                let prog_if = ((class_reg >> 8) & 0xFF) as u8;

                // Match NVMe controller (Class 01h, Subclass 08h, ProgIF 02h)
                if class_code == 0x01 && subclass == 0x08 && prog_if == 0x02 {
                    let device_id = (id_reg >> 16) as u16;

                    // Read BAR0 (offset 0x10)
                    let bar0_lo = pci_read_u32(bus, slot, func, 0x10);
                    let is_64bit = (bar0_lo & 0x06) == 0x04;
                    let bar0_hi = if is_64bit {
                        pci_read_u32(bus, slot, func, 0x14)
                    } else {
                        0
                    };
                    let bar0_phys = ((bar0_hi as u64) << 32) | ((bar0_lo & !0x0F) as u64);

                    // Enable Memory Space (bit 1) and Bus Master (bit 2) in Command Register (offset 0x04)
                    let mut cmd = pci_read_u32(bus, slot, func, 0x04);
                    cmd |= (1 << 1) | (1 << 2);
                    pci_write_u32(bus, slot, func, 0x04, cmd);

                    return Some(PciDeviceInfo {
                        bus,
                        slot,
                        func,
                        vendor_id,
                        device_id,
                        bar0_phys,
                        bar0_mmio: bar0_phys as *mut u8,
                    });
                }

                // If function 0 is not a multi-function device, don't scan func 1..7
                if func == 0 {
                    let header_type = (pci_read_u32(bus, slot, 0, 0x0C) >> 16) & 0xFF;
                    if (header_type & 0x80) == 0 {
                        break;
                    }
                }
            }
        }
    }
    None
}
