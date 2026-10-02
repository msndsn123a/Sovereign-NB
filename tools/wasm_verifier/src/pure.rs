//! Shared, no-std integer inference routines used by both UEFI and WebAssembly.

pub const INPUTS: usize = 64;
pub const HIDDEN: usize = 32;
pub const OUTPUTS: usize = 16;
pub const ATTENTION: usize = 16;
pub const STATE: usize = ATTENTION * ATTENTION;

#[repr(C, align(64))]
pub struct AlignedActivationLut(pub [i16; 256]);

const fn build_silu_lut() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut index = 0usize;
    while index < 256 {
        let x_q4 = index as i32 - 128;
        let abs_x_q4 = if x_q4 < 0 { -x_q4 } else { x_q4 };
        let sigmoid_q8 = 128 + (x_q4 * 128) / (16 + abs_x_q4);
        table[index] = ((x_q4 * sigmoid_q8) / 256) as i16;
        index += 1;
    }
    table
}

pub static SILU_LUT_Q4: AlignedActivationLut = AlignedActivationLut(build_silu_lut());

#[inline(always)]
pub fn silu_q4(input: i8) -> i16 {
    SILU_LUT_Q4.0[(input as i16 + 128) as usize]
}

#[inline]
fn decode_ternary(code: u8) -> i32 {
    match code & 3 {
        1 => 1,
        3 => -1,
        _ => 0,
    }
}

#[inline]
fn weight(packed: &[u8], index: usize) -> i32 {
    decode_ternary(packed[index >> 2] >> ((index & 3) * 2))
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

/// Integer 64 -> 32 -> 16 ternary MLP, matching the UEFI scalar implementation.
pub fn ternary_mlp(
    inputs: &[i8; INPUTS],
    layer1: &[[u8; 16]; HIDDEN],
    layer2: &[[u8; 8]; OUTPUTS],
    activation_lut: bool,
) -> [i32; OUTPUTS] {
    let mut hidden = [0i8; HIDDEN];
    let mut row = 0;
    while row < HIDDEN {
        let mut accumulator = 0i32;
        let mut col = 0;
        while col < INPUTS {
            let code = (layer1[row][col >> 2] >> ((col & 3) * 2)) & 3;
            accumulator += inputs[col] as i32 * decode_ternary(code);
            col += 1;
        }
        hidden[row] = if activation_lut {
            let clipped = accumulator.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            silu_q4(clipped) as i8
        } else {
            hard_sign(accumulator)
        };
        row += 1;
    }
    let mut outputs = [0i32; OUTPUTS];
    row = 0;
    while row < OUTPUTS {
        let mut accumulator = 0i32;
        let mut col = 0;
        while col < HIDDEN {
            let code = (layer2[row][col >> 2] >> ((col & 3) * 2)) & 3;
            accumulator += hidden[col] as i32 * decode_ternary(code);
            col += 1;
        }
        outputs[row] = accumulator;
        row += 1;
    }
    outputs
}

/// Exact integer recurrent causal-attention step matching the UEFI scalar kernel.
pub fn causal_linear_attention(
    inputs: &[i8; INPUTS],
    q_weights: &[[u8; 16]; ATTENTION],
    k_weights: &[[u8; 16]; ATTENTION],
    v_weights: &[[u8; 16]; ATTENTION],
    o_weights: &[[u8; 4]; ATTENTION],
    state: &mut [i32; STATE],
) -> [i32; ATTENTION] {
    let mut q = [0i32; ATTENTION];
    let mut k = [0i32; ATTENTION];
    let mut v = [0i32; ATTENTION];
    let mut row = 0;
    while row < ATTENTION {
        let mut col = 0;
        while col < INPUTS {
            q[row] += inputs[col] as i32 * weight(&q_weights[row], col);
            k[row] += inputs[col] as i32 * weight(&k_weights[row], col);
            v[row] += inputs[col] as i32 * weight(&v_weights[row], col);
            col += 1;
        }
        row += 1;
    }
    row = 0;
    while row < ATTENTION {
        let mut col = 0;
        while col < ATTENTION {
            let index = row * ATTENTION + col;
            state[index] = state[index].saturating_add(k[row].saturating_mul(v[col]));
            col += 1;
        }
        row += 1;
    }
    let mut output = [0i32; ATTENTION];
    row = 0;
    while row < ATTENTION {
        let mut col = 0;
        while col < ATTENTION {
            let index = col * ATTENTION + row;
            output[row] = output[row].wrapping_add(q[col].saturating_mul(state[index]));
            col += 1;
        }
        row += 1;
    }
    let mut projected = [0i32; ATTENTION];
    row = 0;
    while row < ATTENTION {
        let mut col = 0;
        while col < ATTENTION {
            let code = (o_weights[row][col >> 2] >> ((col & 3) * 2)) & 3;
            projected[row] =
                projected[row].wrapping_add(output[col].saturating_mul(decode_ternary(code)));
            col += 1;
        }
        row += 1;
    }
    projected
}
