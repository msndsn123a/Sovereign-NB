#![cfg_attr(not(windows), allow(dead_code))]

#[cfg(not(windows))]
compile_error!("mmio_injector is a Windows-only IVSHMEM producer");

#[cfg(windows)]
mod windows_host {
    use core::ffi::c_void;
    use core::ptr;
    use core::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    type Handle = *mut c_void;
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const FILE_SHARE_WRITE: u32 = 2;
    const OPEN_EXISTING: u32 = 3;
    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0x000F_001F;
    const MAILBOX_MAGIC: u32 = 0x5348_4D42;
    const MAILBOX_VERSION: u32 = 1;
    const CAPACITY: u64 = 16;
    const INPUT_READY: u32 = 1;
    const OUTPUT_READY: u32 = 1;
    const INPUT_BASE: usize = 64;
    const INPUT_SLOT_SIZE: usize = 128;
    const OUTPUT_BASE: usize = INPUT_BASE + 16 * INPUT_SLOT_SIZE;
    const OUTPUT_SLOT_SIZE: usize = 384;
    const OUTPUT_VALUES_OFFSET: usize = 64;
    const OUTPUT_METADATA_OFFSET: usize = 320;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            creation: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn CreateFileMappingW(
            file: Handle,
            security: *mut c_void,
            protection: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(address: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    struct Mapping {
        file: Handle,
        mapping: Handle,
        view: *mut u8,
    }

    impl Mapping {
        fn open(path: &str) -> Result<Self, String> {
            use std::os::windows::ffi::OsStrExt;
            let wide: Vec<u16> = std::ffi::OsStr::new(path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            unsafe {
                let file = CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    ptr::null_mut(),
                    OPEN_EXISTING,
                    0,
                    ptr::null_mut(),
                );
                if file as isize == -1 {
                    return Err(format!("CreateFileW failed: {}", std::io::Error::last_os_error()));
                }
                let mapping = CreateFileMappingW(
                    file,
                    ptr::null_mut(),
                    PAGE_READWRITE,
                    0,
                    0,
                    ptr::null(),
                );
                if mapping.is_null() {
                    CloseHandle(file);
                    return Err(format!("CreateFileMappingW failed: {}", std::io::Error::last_os_error()));
                }
                let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0).cast::<u8>();
                if view.is_null() {
                    CloseHandle(mapping);
                    CloseHandle(file);
                    return Err(format!("MapViewOfFile failed: {}", std::io::Error::last_os_error()));
                }
                Ok(Self { file, mapping, view })
            }
        }

        fn atomic_u32(&self, offset: usize) -> &AtomicU32 {
            assert_eq!((self.view as usize + offset) % 4, 0);
            unsafe { &*self.view.add(offset).cast::<AtomicU32>() }
        }

        fn atomic_u64(&self, offset: usize) -> &AtomicU64 {
            assert_eq!((self.view as usize + offset) % 8, 0);
            unsafe { &*self.view.add(offset).cast::<AtomicU64>() }
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(self.view.cast());
                CloseHandle(self.mapping);
                CloseHandle(self.file);
            }
        }
    }

    fn wait_for<T, F: Fn() -> Option<T>>(timeout: Duration, poll: F) -> Result<T, String> {
        let start = Instant::now();
        loop {
            if let Some(value) = poll() {
                return Ok(value);
            }
            if start.elapsed() >= timeout {
                return Err("timed out waiting for IVSHMEM mailbox transition".to_string());
            }
            std::hint::spin_loop();
        }
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args().skip(1);
        let path = args.next().ok_or("usage: mmio_injector <backing-file> [frame-count]")?;
        let frames = args
            .next()
            .unwrap_or_else(|| "8".to_string())
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if frames == 0 || frames > CAPACITY as usize {
            return Err("frame count must be 1..=16".to_string());
        }

        let mapping = Mapping::open(&path)?;
        wait_for(Duration::from_secs(30), || unsafe {
            let magic = ptr::read_volatile(mapping.view.cast::<u32>());
            let version = ptr::read_volatile(mapping.view.add(4).cast::<u32>());
            (magic == MAILBOX_MAGIC && version == MAILBOX_VERSION).then_some(())
        })?;
        fence(Ordering::Acquire);
        println!("[MMIO HOST]: IVSHMEM mailbox ready magic=0x{MAILBOX_MAGIC:08X}, version={MAILBOX_VERSION}");

        let mut host_latencies = [Duration::ZERO; 16];
        for frame in 0..frames {
            let input_head = mapping.atomic_u64(8);
            let input_tail = mapping.atomic_u64(16);
            let sequence = input_head.load(Ordering::Relaxed);
            if sequence.wrapping_sub(input_tail.load(Ordering::Acquire)) >= CAPACITY {
                return Err("ingress ring full before publish".to_string());
            }
            let input_offset = INPUT_BASE + (sequence as usize % CAPACITY as usize) * INPUT_SLOT_SIZE;
            let input_state = mapping.atomic_u32(input_offset);
            if input_state.load(Ordering::Acquire) != 0 {
                return Err("selected ingress slot was not empty".to_string());
            }

            let started = Instant::now();
            unsafe {
                ptr::write_volatile(mapping.view.add(input_offset + 8).cast::<u64>(), 0);
                for index in 0..64 {
                    ptr::write_volatile(mapping.view.add(input_offset + 64 + index), 0);
                }
            }
            input_state.store(INPUT_READY, Ordering::Release);
            input_head.store(sequence.wrapping_add(1), Ordering::Release);

            let (output_sequence, output_offset) = wait_for(Duration::from_secs(10), || {
                let tail = mapping.atomic_u64(32).load(Ordering::Relaxed);
                let head = mapping.atomic_u64(24).load(Ordering::Acquire);
                if head == tail {
                    return None;
                }
                let offset = OUTPUT_BASE + (tail as usize % CAPACITY as usize) * OUTPUT_SLOT_SIZE;
                (mapping.atomic_u32(offset).load(Ordering::Acquire) == OUTPUT_READY)
                    .then_some((tail, offset))
            })?;

            let output_dim = unsafe {
                ptr::read_volatile(
                    mapping
                        .view
                        .add(output_offset + OUTPUT_METADATA_OFFSET)
                        .cast::<u32>(),
                )
            };
            if output_dim != 16 {
                return Err(format!("frame {frame}: output_dim={output_dim}, expected 16"));
            }
            for index in 0..16 {
                let value = unsafe {
                    ptr::read_unaligned(
                        mapping
                            .view
                            .add(output_offset + OUTPUT_VALUES_OFFSET + index * 4)
                            .cast::<i32>(),
                    )
                };
                if value != 0 {
                    return Err(format!("frame {frame}, output {index}: got {value}, expected 0"));
                }
            }

            mapping.atomic_u32(output_offset).store(0, Ordering::Release);
            mapping
                .atomic_u64(32)
                .store(output_sequence.wrapping_add(1), Ordering::Release);
            host_latencies[frame] = started.elapsed();
            println!(
                "[MMIO HOST]: frame={frame} output=match round_trip_us={:.3}",
                host_latencies[frame].as_secs_f64() * 1_000_000.0
            );
        }
        println!("[MMIO HOST]: frames={frames}, parity=100%, packet_loss=0");
        Ok(())
    }
}

fn main() {
    if let Err(error) = windows_host::run() {
        eprintln!("[MMIO HOST ERROR]: {error}");
        std::process::exit(1);
    }
}
