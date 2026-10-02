//! Polling-Driven Bare-Metal NVMe Driver.
//!
//! Provides zero-copy DMA block ingestion from NVMe storage controllers
//! without OS drivers, interrupts, or runtime allocators.

use crate::pci::PciDeviceInfo;
use core::ptr::{read_volatile, write_volatile};

pub const QUEUE_SIZE: usize = 64;
pub const DMA_PAGE_SIZE: usize = 4096;

#[repr(C, align(4096))]
pub struct IdentifyBuffer(pub [u8; DMA_PAGE_SIZE]);

static mut NAMESPACE_IDENTIFY_BUFFER: IdentifyBuffer = IdentifyBuffer([0; DMA_PAGE_SIZE]);

/// NVMe 64-byte Submission Queue Entry (SQE).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct NvmeSqEntry {
    pub cdw0: u32,
    pub nsid: u32,
    pub cdw2: u32,
    pub cdw3: u32,
    pub mptr: u64,
    pub prp1: u64,
    pub prp2: u64,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

impl NvmeSqEntry {
    pub const ZERO: Self = Self {
        cdw0: 0,
        nsid: 0,
        cdw2: 0,
        cdw3: 0,
        mptr: 0,
        prp1: 0,
        prp2: 0,
        cdw10: 0,
        cdw11: 0,
        cdw12: 0,
        cdw13: 0,
        cdw14: 0,
        cdw15: 0,
    };
}

/// NVMe 16-byte Completion Queue Entry (CQE).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NvmeCqEntry {
    pub cdw0: u32,
    pub rsvd: u32,
    pub sq_head: u16,
    pub sq_id: u16,
    pub cid: u16,
    pub status: u16,
}

impl NvmeCqEntry {
    pub const ZERO: Self = Self {
        cdw0: 0,
        rsvd: 0,
        sq_head: 0,
        sq_id: 0,
        cid: 0,
        status: 0,
    };
}

/// 4096-byte memory-page aligned circular queue buffers (required by NVMe spec for ASQ/ACQ).
#[repr(C, align(4096))]
pub struct AlignedSq(pub [NvmeSqEntry; QUEUE_SIZE]);

#[repr(C, align(4096))]
pub struct AlignedCq(pub [NvmeCqEntry; QUEUE_SIZE]);

pub static mut ADMIN_SQ: AlignedSq = AlignedSq([NvmeSqEntry::ZERO; QUEUE_SIZE]);
pub static mut ADMIN_CQ: AlignedCq = AlignedCq([NvmeCqEntry::ZERO; QUEUE_SIZE]);
pub static mut IO_SQ: AlignedSq = AlignedSq([NvmeSqEntry::ZERO; QUEUE_SIZE]);
pub static mut IO_CQ: AlignedCq = AlignedCq([NvmeCqEntry::ZERO; QUEUE_SIZE]);

/// Active NVMe Controller handle.
pub struct NvmeController {
    pub bar0: *mut u8,
    pub db_stride: usize,
    pub logical_block_size: usize,
    pub namespace_size_blocks: u64,
    pub asq_tail: u16,
    pub acq_head: u16,
    pub acq_phase: u16,
    pub iosq_tail: u16,
    pub iocq_head: u16,
    pub iocq_phase: u16,
    pub next_cid: u16,
}

impl NvmeController {
    /// Initializes the NVMe controller via MMIO, sets up admin queues, enables the controller,
    /// and constructs the primary I/O submission and completion queues.
    pub unsafe fn init(pci_dev: &PciDeviceInfo) -> Result<Self, &'static str> {
        let bar0 = pci_dev.bar0_mmio;

        // 1. Read Controller Capabilities (CAP) at offset 0x00
        let cap = read_volatile((bar0 as *const u64).add(0));
        let dstrd = ((cap >> 32) & 0x0F) as usize;
        let db_stride = 4 << dstrd;
        let to_500ms = ((cap >> 24) & 0xFF) as u32;
        let mpsmin = ((cap >> 48) & 0x0F) as u8;
        if mpsmin != 0 {
            return Err("NVMe controller requires a memory page larger than 4 KiB");
        }

        crate::serial_println!(
            "[NVMe DEBUG] BAR0: {:p}, CAP: {:#018x}, DSTRD: {}, TO: {}x500ms",
            bar0,
            cap,
            dstrd,
            to_500ms
        );

        let mut ctrl = Self {
            bar0,
            db_stride,
            logical_block_size: 0,
            namespace_size_blocks: 0,
            asq_tail: 0,
            acq_head: 0,
            acq_phase: 1,
            iosq_tail: 0,
            iocq_head: 0,
            iocq_phase: 1,
            next_cid: 1,
        };

        // 2. Disable controller (CC.EN = 0) if currently enabled
        let cc_ptr = (bar0.add(0x14)) as *mut u32;
        let csts_ptr = (bar0.add(0x1C)) as *const u32;

        let mut cc = read_volatile(cc_ptr);
        let mut csts = read_volatile(csts_ptr);
        crate::serial_println!(
            "[NVMe DEBUG] Initial CC: {:#010x}, CSTS: {:#010x}",
            cc,
            csts
        );

        if (cc & 1) != 0 {
            write_volatile(cc_ptr, cc & !1);
        }

        // Wait until CSTS.RDY == 0
        let mut timeout = 10_000_000;
        while (read_volatile(csts_ptr) & 1) != 0 {
            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 {
                return Err("Timeout waiting for NVMe CSTS.RDY == 0");
            }
        }

        // 3. Configure Admin Queue Attributes (AQA at offset 0x24)
        // ASQS (Admin SQ Size) in bits 11:0, ACQS (Admin CQ Size) in bits 27:16 (0-based)
        let aqa_val = (((QUEUE_SIZE - 1) as u32) << 16) | ((QUEUE_SIZE - 1) as u32);
        write_volatile((bar0.add(0x24)) as *mut u32, aqa_val);

        // 4. Configure Admin SQ and CQ Base Addresses (ASQ at 0x28, ACQ at 0x30)
        let asq_phys = core::ptr::addr_of!(ADMIN_SQ.0) as u64;
        let acq_phys = core::ptr::addr_of!(ADMIN_CQ.0) as u64;
        write_volatile((bar0.add(0x28)) as *mut u64, asq_phys);
        write_volatile((bar0.add(0x30)) as *mut u64, acq_phys);

        crate::serial_println!(
            "[NVMe DEBUG] AQA: {:#010x}, ASQ: {:#018x}, ACQ: {:#018x}",
            aqa_val,
            asq_phys,
            acq_phys
        );

        // 5. Enable controller (CC at 0x14)
        // IOSQES = 6 (64-byte SQE), IOCQES = 4 (16-byte CQE), MPS = 0 (4KiB page), CSS = 0 (NVM), EN = 1
        cc = (6 << 16) | (4 << 20) | 1;
        write_volatile(cc_ptr, cc);

        // Wait until CSTS.RDY == 1
        timeout = 10_000_000;
        loop {
            csts = read_volatile(csts_ptr);
            if (csts & 1) == 1 {
                break;
            }
            if (csts & 2) != 0 {
                crate::serial_println!(
                    "[NVMe DEBUG] CSTS CFS (Fatal Status) set! CSTS: {:#010x}",
                    csts
                );
                return Err("NVMe CSTS.CFS Fatal Status set");
            }
            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 {
                crate::serial_println!("[NVMe DEBUG] CSTS timeout! CSTS: {:#010x}", csts);
                return Err("Timeout waiting for NVMe CSTS.RDY == 1");
            }
        }

        // 6. Create I/O Completion Queue (Admin command 0x05)
        let iocq_phys = core::ptr::addr_of!(IO_CQ.0) as u64;
        let mut create_cq_cmd = NvmeSqEntry::ZERO;
        create_cq_cmd.cdw0 = 0x05; // Create I/O Completion Queue
        create_cq_cmd.prp1 = iocq_phys;
        // CDW10: QSIZE (bits 31:16) = QUEUE_SIZE - 1, QID (bits 15:0) = 1
        create_cq_cmd.cdw10 = (((QUEUE_SIZE - 1) as u32) << 16) | 1;
        // CDW11: Physically Contiguous (bit 0 = 1), Interrupts Disabled (bit 1 = 0)
        create_cq_cmd.cdw11 = 0x0001;
        ctrl.submit_admin_cmd(&create_cq_cmd)?;

        // 7. Create I/O Submission Queue (Admin command 0x01)
        let iosq_phys = core::ptr::addr_of!(IO_SQ.0) as u64;
        let mut create_sq_cmd = NvmeSqEntry::ZERO;
        create_sq_cmd.cdw0 = 0x01; // Create I/O Submission Queue
        create_sq_cmd.prp1 = iosq_phys;
        // CDW10: QSIZE (bits 31:16) = QUEUE_SIZE - 1, QID (bits 15:0) = 1
        create_sq_cmd.cdw10 = (((QUEUE_SIZE - 1) as u32) << 16) | 1;
        // CDW11: CQID (bits 31:16) = 1, Physically Contiguous (bit 0 = 1)
        create_sq_cmd.cdw11 = (1 << 16) | 0x0001;
        ctrl.submit_admin_cmd(&create_sq_cmd)?;

        // 8. Identify Namespace 1 to inspect logical block size
        let id_buf = core::ptr::addr_of_mut!(NAMESPACE_IDENTIFY_BUFFER.0) as *mut u8;
        core::ptr::write_bytes(id_buf, 0, DMA_PAGE_SIZE);
        let mut id_cmd = NvmeSqEntry::ZERO;
        id_cmd.cdw0 = 0x06; // Identify
        id_cmd.nsid = 1;
        id_cmd.prp1 = id_buf as u64;
        id_cmd.cdw10 = 0x00; // CNS 0 = Namespace
        ctrl.submit_admin_cmd(&id_cmd)?;

        let nsze = core::ptr::read_unaligned(id_buf as *const u64);
        let flbas = *id_buf.add(26);
        let lbaf_idx = (flbas & 0x0F) as usize;
        let nlbaf = *id_buf.add(25) as usize;
        if lbaf_idx > nlbaf {
            return Err("NVMe namespace selected an unavailable LBA format");
        }
        let lbaf_offset = 128 + lbaf_idx * 4;
        let metadata_size = core::ptr::read_unaligned(id_buf.add(lbaf_offset) as *const u16);
        let lbads = *id_buf.add(lbaf_offset + 2);
        let logical_block_size = 1usize
            .checked_shl(lbads as u32)
            .filter(|size| (512..=DMA_PAGE_SIZE).contains(size))
            .ok_or("NVMe namespace LBA size is unsupported")?;
        if nsze == 0 {
            return Err("NVMe namespace has zero blocks");
        }
        ctrl.logical_block_size = logical_block_size;
        ctrl.namespace_size_blocks = nsze;
        crate::serial_println!(
            "[NVMe]: Namespace 1: NSZE={}, FLBAS={}, LBADS={}, LBA_SIZE={} bytes, MS={}",
            nsze,
            flbas,
            lbads,
            logical_block_size,
            metadata_size
        );

        Ok(ctrl)
    }

    /// Submits a command to the Admin Submission Queue and busy-polls the Admin CQ for completion.
    pub unsafe fn submit_admin_cmd(
        &mut self,
        cmd: &NvmeSqEntry,
    ) -> Result<NvmeCqEntry, &'static str> {
        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1);

        let mut entry = *cmd;
        entry.cdw0 |= (cid as u32) << 16;

        let asq_idx = self.asq_tail as usize;
        write_volatile(core::ptr::addr_of_mut!(ADMIN_SQ.0[asq_idx]), entry);

        self.asq_tail = ((self.asq_tail + 1) as usize % QUEUE_SIZE) as u16;

        // Ring Admin SQ Tail Doorbell (offset 0x1000)
        let sq_db = self.bar0.add(0x1000) as *mut u32;
        write_volatile(sq_db, self.asq_tail as u32);

        // Poll Admin CQ for completion entry
        let mut timeout = 10_000_000;
        let acq_idx = self.acq_head as usize;
        loop {
            let cqe = read_volatile(core::ptr::addr_of!(ADMIN_CQ.0[acq_idx]));
            let phase = cqe.status & 1;
            if phase == self.acq_phase {
                // Advance ACQ head
                self.acq_head = ((self.acq_head + 1) as usize % QUEUE_SIZE) as u16;
                if self.acq_head == 0 {
                    self.acq_phase ^= 1;
                }

                // Ring Admin CQ Head Doorbell (offset 0x1000 + db_stride)
                let cq_db = self.bar0.add(0x1000 + self.db_stride) as *mut u32;
                write_volatile(cq_db, self.acq_head as u32);

                let status_code = (cqe.status >> 1) & 0x7FFF;
                if status_code != 0 {
                    return Err("Admin command returned non-zero status");
                }
                return Ok(cqe);
            }

            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 {
                return Err("Timeout waiting for Admin command completion");
            }
        }
    }

    /// Issues an NVMe Read Command (opcode 0x02) to the I/O Submission Queue
    /// directly transferring `num_blocks` starting at `lba_start` into `dest_phys_addr` via DMA.
    pub unsafe fn read_raw_lba(
        &mut self,
        lba_start: u64,
        num_blocks: u16,
        dest_phys_addr: *mut u8,
        dest_capacity: usize,
    ) -> Result<(), &'static str> {
        if num_blocks == 0 {
            return Err("NVMe read block count must be nonzero");
        }
        if self.logical_block_size == 0 {
            return Err("NVMe namespace block size has not been identified");
        }
        if (dest_phys_addr as usize) & (DMA_PAGE_SIZE - 1) != 0 {
            return Err("NVMe DMA destination must be 4 KiB aligned");
        }
        let byte_count = (num_blocks as usize)
            .checked_mul(self.logical_block_size)
            .ok_or("NVMe DMA transfer size overflow")?;
        if byte_count > DMA_PAGE_SIZE || byte_count > dest_capacity {
            return Err("NVMe DMA transfer exceeds the aligned destination page");
        }
        if lba_start >= self.namespace_size_blocks
            || num_blocks as u64 > self.namespace_size_blocks - lba_start
        {
            return Err("NVMe read exceeds namespace bounds");
        }

        let cid = self.next_cid;
        self.next_cid = self.next_cid.wrapping_add(1);

        let mut cmd = NvmeSqEntry::ZERO;
        cmd.cdw0 = 0x02 | ((cid as u32) << 16); // Opcode 0x02 = Read
        cmd.nsid = 1; // Primary Namespace 1
        cmd.prp1 = dest_phys_addr as u64; // Direct DMA physical address destination
        cmd.cdw10 = (lba_start & 0xFFFFFFFF) as u32;
        cmd.cdw11 = (lba_start >> 32) as u32;
        cmd.cdw12 = (num_blocks - 1) as u32; // 0-based count

        let iosq_idx = self.iosq_tail as usize;
        write_volatile(core::ptr::addr_of_mut!(IO_SQ.0[iosq_idx]), cmd);

        self.iosq_tail = ((self.iosq_tail + 1) as usize % QUEUE_SIZE) as u16;

        // Ring I/O SQ1 Tail Doorbell: offset 0x1000 + 2 * db_stride
        let sq_db = self.bar0.add(0x1000 + 2 * self.db_stride) as *mut u32;
        write_volatile(sq_db, self.iosq_tail as u32);

        // Poll I/O CQ1 for completion
        let mut timeout = 10_000_000;
        let iocq_idx = self.iocq_head as usize;
        loop {
            let cqe = read_volatile(core::ptr::addr_of!(IO_CQ.0[iocq_idx]));
            let phase = cqe.status & 1;
            if phase == self.iocq_phase {
                // Advance IOCQ head
                self.iocq_head = ((self.iocq_head + 1) as usize % QUEUE_SIZE) as u16;
                if self.iocq_head == 0 {
                    self.iocq_phase ^= 1;
                }

                // Ring I/O CQ1 Head Doorbell: offset 0x1000 + 3 * db_stride
                let cq_db = self.bar0.add(0x1000 + 3 * self.db_stride) as *mut u32;
                write_volatile(cq_db, self.iocq_head as u32);

                let status_code = (cqe.status >> 1) & 0x7FFF;
                if status_code != 0 {
                    return Err("NVMe Read command returned non-zero error status");
                }
                return Ok(());
            }

            core::hint::spin_loop();
            timeout -= 1;
            if timeout == 0 {
                return Err("Timeout waiting for NVMe Read DMA completion");
            }
        }
    }
}
