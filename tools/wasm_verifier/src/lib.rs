#![cfg_attr(not(feature = "std"), no_std)]

mod pure;
pub use pure::{causal_linear_attention, silu_q4, ternary_mlp, AlignedActivationLut, SILU_LUT_Q4};

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn neur_silu_lut(input: i32) -> i32 {
    silu_q4(input.clamp(-128, 127) as i8) as i32
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn neur_mlp(
    inputs: *const i8,
    layer1: *const u8,
    layer2: *const u8,
    activation_lut: u32,
    output: *mut i32,
) -> i32 {
    if inputs.is_null() || layer1.is_null() || layer2.is_null() || output.is_null() {
        return -1;
    }
    let inputs = &*(inputs as *const [i8; 64]);
    let layer1 = &*(layer1 as *const [[u8; 16]; 32]);
    let layer2 = &*(layer2 as *const [[u8; 8]; 16]);
    let result = ternary_mlp(inputs, layer1, layer2, activation_lut != 0);
    core::ptr::copy_nonoverlapping(result.as_ptr(), output, 16);
    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn neur_causal_attention(
    inputs: *const i8,
    packed_weights: *const u8,
    state: *mut i32,
    output: *mut i32,
) -> i32 {
    if inputs.is_null() || packed_weights.is_null() || state.is_null() || output.is_null() {
        return -1;
    }
    let inputs = &*(inputs as *const [i8; 64]);
    let q = &*(packed_weights as *const [[u8; 16]; 16]);
    let k = &*((packed_weights.add(256)) as *const [[u8; 16]; 16]);
    let v = &*((packed_weights.add(512)) as *const [[u8; 16]; 16]);
    let o = &*((packed_weights.add(768)) as *const [[u8; 4]; 16]);
    let state = &mut *(state as *mut [i32; 256]);
    let result = causal_linear_attention(inputs, q, k, v, o, state);
    core::ptr::copy_nonoverlapping(result.as_ptr(), output, 16);
    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn neur_alloc(size: usize, alignment: usize) -> *mut u8 {
    extern "C" {
        static __heap_base: u8;
    }
    static mut NEXT: usize = 0;
    let alignment = alignment.max(1).next_power_of_two();
    unsafe {
        let start = if NEXT == 0 {
            &__heap_base as *const u8 as usize
        } else {
            NEXT
        };
        let aligned = (start + alignment - 1) & !(alignment - 1);
        let end = aligned.saturating_add(size);
        let current_pages = core::arch::wasm32::memory_size(0);
        let required_pages = end.div_ceil(65536);
        if required_pages > current_pages {
            let delta = required_pages - current_pages;
            if core::arch::wasm32::memory_grow(0, delta) == usize::MAX {
                return core::ptr::null_mut();
            }
        }
        NEXT = end;
        aligned as *mut u8
    }
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_ternary(packed: &mut [u8], weight_index: usize, value: i8) {
        let code = if value > 0 {
            1
        } else if value < 0 {
            3
        } else {
            0
        };
        let byte = weight_index >> 2;
        let shift = (weight_index & 3) * 2;
        packed[byte] = (packed[byte] & !(3 << shift)) | (code << shift);
    }

    #[test]
    fn mlp_matches_integer_reference_vectors() {
        let mut layer1 = [[0u8; 16]; 32];
        let mut layer2 = [[0u8; 8]; 16];
        set_ternary(&mut layer1[0], 0, 1);
        set_ternary(&mut layer1[1], 1, -1);
        set_ternary(&mut layer2[0], 0, 1);
        set_ternary(&mut layer2[1], 1, 1);
        let mut input = [0i8; 64];
        input[0] = 5;
        input[1] = 7;
        let output = ternary_mlp(&input, &layer1, &layer2, false);
        assert_eq!(&output[..2], &[1, -1]);
        assert!(output[2..].iter().all(|value| *value == 0));
    }

    #[test]
    fn silu_lut_is_exact_integer_lookup() {
        assert_eq!(silu_q4(-3), -1);
        assert_eq!(silu_q4(0), 0);
        assert_eq!((SILU_LUT_Q4.0.as_ptr() as usize) & 63, 0);
    }

    #[test]
    fn causal_attention_accumulates_state_exactly() {
        let mut q = [[0u8; 16]; 16];
        let mut k = [[0u8; 16]; 16];
        let mut v = [[0u8; 16]; 16];
        let mut o = [[0u8; 4]; 16];
        set_ternary(&mut q[0], 0, 1);
        set_ternary(&mut k[0], 1, 1);
        set_ternary(&mut v[0], 2, 1);
        set_ternary(&mut o[0], 0, 1);
        let mut state = [0i32; 256];
        let mut input = [0i8; 64];
        input[0] = 2;
        input[1] = 3;
        input[2] = 4;
        assert_eq!(
            causal_linear_attention(&input, &q, &k, &v, &o, &mut state)[0],
            24
        );
        input[0] = 1;
        input[1] = 2;
        input[2] = 3;
        assert_eq!(
            causal_linear_attention(&input, &q, &k, &v, &o, &mut state)[0],
            18
        );
        assert_eq!(state[0], 18);
    }
}
