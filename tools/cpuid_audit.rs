use std::arch::x86_64::{__cpuid, __cpuid_count};

fn main() {
    let leaf0 = __cpuid(0);
    let vendor = [leaf0.ebx, leaf0.edx, leaf0.ecx]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let vendor = String::from_utf8_lossy(&vendor);

    let max_extended = __cpuid(0x8000_0000).eax;
    let mut brand_bytes = Vec::with_capacity(48);
    if max_extended >= 0x8000_0004 {
        for leaf in 0x8000_0002..=0x8000_0004 {
            let result = __cpuid(leaf);
            for word in [result.eax, result.ebx, result.ecx, result.edx] {
                brand_bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    let brand = String::from_utf8_lossy(&brand_bytes).trim().to_string();

    let leaf1 = __cpuid(1);
    let hypervisor_present = (leaf1.ecx & (1 << 31)) != 0;
    let xsave = (leaf1.ecx & (1 << 26)) != 0;
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;
    let max_basic = leaf0.eax;
    let leaf7 = if max_basic >= 7 {
        __cpuid_count(7, 0)
    } else {
        __cpuid(0)
    };
    let avx512f = max_basic >= 7 && leaf7.ebx & (1 << 16) != 0;
    let avx512dq = max_basic >= 7 && leaf7.ebx & (1 << 17) != 0;
    let avx512bw = max_basic >= 7 && leaf7.ebx & (1 << 30) != 0;
    let avx512vl = max_basic >= 7 && leaf7.ebx & (1 << 31) != 0;

    let xcr0 = if xsave && osxsave {
        let low: u32;
        let high: u32;
        unsafe {
            core::arch::asm!(
                "xgetbv",
                in("ecx") 0u32,
                out("eax") low,
                out("edx") high,
                options(nomem, nostack, preserves_flags),
            );
        }
        ((high as u64) << 32) | low as u64
    } else {
        0
    };
    let zmm_state_enabled = xcr0 & 0xE6 == 0xE6;

    println!("[HOST CPUID]: vendor={vendor}, brand={brand}");
    println!("[HOST CPUID]: hypervisor_present={hypervisor_present}");
    println!("[HOST CPUID]: XSAVE={xsave}, OSXSAVE={osxsave}, XCR0=0x{xcr0:016X}, AVX512_state_enabled={zmm_state_enabled}");
    println!("[HOST CPUID]: AVX512F={avx512f}");
    println!("[HOST CPUID]: AVX512BW={avx512bw}");
    println!("[HOST CPUID]: AVX512DQ={avx512dq}");
    println!("[HOST CPUID]: AVX512VL={avx512vl}");
    println!("[HOST CPUID]: host_avx512_kernel_requirements_met={}", avx512f && avx512bw && avx512dq && avx512vl && zmm_state_enabled);
}