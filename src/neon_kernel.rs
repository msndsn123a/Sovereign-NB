//! AArch64 NEON implementation of the integer 64 -> 32 -> 16 ternary MLP.
//!
//! Weights are expanded to signed bytes in small fixed stack buffers, then
//! reduced with widening NEON multiplies. The shared scalar kernel remains the
//! canonical reference and the activation path is the same integer SiLU LUT.

use core::arch::aarch64::{vaddvq_s32, vld1_s8, vmull_s8, vpaddlq_s16};

const INPUTS: usize = 64;
const HIDDEN: usize = 32;
const OUTPUTS: usize = 16;

#[inline]
fn decode(code: u8) -> i8 {
    match code & 3 {
        1 => 1,
        3 => -1,
        _ => 0,
    }
}

#[inline]
fn hard_sign(value: i32) -> i8 {
    if value < 0 {
        -1
    } else if value > 0 {
        1
    } else {
        0
    }
}

#[inline]
unsafe fn dot_neon(input: &[i8; INPUTS], weights: &[i8; INPUTS]) -> i32 {
    let mut sum = 0i32;
    let mut offset = 0usize;
    while offset < INPUTS {
        let input_vector = vld1_s8(input.as_ptr().add(offset));
        let weight_vector = vld1_s8(weights.as_ptr().add(offset));
        let products = vmull_s8(input_vector, weight_vector);
        let pairs = vpaddlq_s16(products);
        sum += vaddvq_s32(pairs);
        offset += 8;
    }
    sum
}

/// Exact integer NEON MLP; rows use the same packed ternary format as NEUR v2.
#[target_feature(enable = "neon")]
pub unsafe fn ternary_mlp_neon(
    inputs: &[i8; INPUTS],
    layer1: &[[u8; 16]; HIDDEN],
    layer2: &[[u8; 8]; OUTPUTS],
    activation_lut: bool,
) -> [i32; OUTPUTS] {
    let mut hidden = [0i8; HIDDEN];
    let mut row = 0usize;
    while row < HIDDEN {
        let mut weights = [0i8; INPUTS];
        let mut column = 0usize;
        while column < INPUTS {
            weights[column] = decode(layer1[row][column >> 2] >> ((column & 3) * 2));
            column += 1;
        }
        let accumulator = dot_neon(inputs, &weights);
        hidden[row] = if activation_lut {
            let clipped = accumulator.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            crate::wasm_math::silu_q4(clipped) as i8
        } else {
            hard_sign(accumulator)
        };
        row += 1;
    }

    let mut outputs = [0i32; OUTPUTS];
    row = 0;
    while row < OUTPUTS {
        let mut input_vector = [0i8; INPUTS];
        input_vector[..HIDDEN].copy_from_slice(&hidden);
        let mut weights = [0i8; INPUTS];
        let mut column = 0usize;
        while column < HIDDEN {
            weights[column] = decode(layer2[row][column >> 2] >> ((column & 3) * 2));
            column += 1;
        }
        outputs[row] = dot_neon(&input_vector, &weights);
        row += 1;
    }
    outputs
}

/// Build deterministic model/input vectors and require exact scalar parity.
pub fn parity_self_test() -> bool {
    let mut layer1 = [[0u8; 16]; HIDDEN];
    let mut layer2 = [[0u8; 8]; OUTPUTS];
    let mut row = 0usize;
    while row < HIDDEN {
        let mut column = 0usize;
        while column < INPUTS {
            let code = match (row * 13 + column * 7) % 3 {
                0 => 0,
                1 => 1,
                _ => 3,
            };
            layer1[row][column >> 2] |= code << ((column & 3) * 2);
            column += 1;
        }
        row += 1;
    }
    row = 0;
    while row < OUTPUTS {
        let mut column = 0usize;
        while column < HIDDEN {
            let code = match (row * 5 + column * 11) % 3 {
                0 => 0,
                1 => 1,
                _ => 3,
            };
            layer2[row][column >> 2] |= code << ((column & 3) * 2);
            column += 1;
        }
        row += 1;
    }

    let mut vectors = [[0i8; INPUTS]; 3];
    let mut column = 0usize;
    while column < INPUTS {
        vectors[0][column] = (column.wrapping_mul(73).wrapping_add(19) as u8) as i8;
        vectors[1][column] = if column & 1 == 0 { i8::MIN } else { i8::MAX };
        vectors[2][column] = match column & 3 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => i8::MIN,
        };
        column += 1;
    }

    let mut vector = 0usize;
    while vector < vectors.len() {
        let mut activation = false;
        while !activation {
            let scalar =
                crate::wasm_math::ternary_mlp(&vectors[vector], &layer1, &layer2, activation);
            let neon = unsafe { ternary_mlp_neon(&vectors[vector], &layer1, &layer2, activation) };
            if scalar != neon {
                return false;
            }
            activation = true;
        }
        let scalar = crate::wasm_math::ternary_mlp(&vectors[vector], &layer1, &layer2, true);
        let neon = unsafe { ternary_mlp_neon(&vectors[vector], &layer1, &layer2, true) };
        if scalar != neon {
            return false;
        }
        vector += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn neon_matches_scalar_reference_for_extreme_vectors_and_activations() {
        assert!(super::parity_self_test());
    }
}
