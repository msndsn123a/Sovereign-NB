//! Fixed-buffer Ed25519 verification for signed NEUR model shards.
//!
//! The firmware enables ed25519-dalek without its `std` or `alloc` features.
//! The embedded key is a deterministic development/test key (RFC 8032 test
//! vector 1); production firmware should replace it with a project-controlled
//! public key and use an externally protected private signing key.

use ed25519_dalek::{Signature, VerifyingKey};

pub const NEUR_SIGNING_PUBLIC_KEY: [u8; 32] = [
    0xD7, 0x5A, 0x98, 0x01, 0x82, 0xB1, 0x0A, 0xB7, 0xD5, 0x4B, 0xFE, 0xD3, 0xC9, 0x64, 0x07, 0x3A,
    0x0E, 0xE1, 0x72, 0xF3, 0xDA, 0xA6, 0x23, 0x25, 0xAF, 0x02, 0x1A, 0x68, 0xF7, 0x07, 0x51, 0x1A,
];

const MAX_NEUR_PAYLOAD_SIZE: usize = 1664;
const BASE_HEADER_SIZE: usize = 16;

/// Verify Ed25519 over `metadata_prefix || payload`, excluding the signature field.
///
/// This fixed-size message buffer is bounded by the NEUR format and does not
/// use `alloc`; the signature and public key are also fixed-size values.
pub fn verify_neur_signature(
    metadata_prefix: &[u8],
    signature_bytes: &[u8],
    payload: &[u8],
) -> bool {
    if metadata_prefix.len() != BASE_HEADER_SIZE
        || signature_bytes.len() != 64
        || payload.is_empty()
        || payload.len() > MAX_NEUR_PAYLOAD_SIZE
    {
        return false;
    }

    let Ok(verifying_key) = VerifyingKey::from_bytes(&NEUR_SIGNING_PUBLIC_KEY) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(signature_bytes) else {
        return false;
    };

    let mut message = [0u8; BASE_HEADER_SIZE + MAX_NEUR_PAYLOAD_SIZE];
    message[..BASE_HEADER_SIZE].copy_from_slice(metadata_prefix);
    message[BASE_HEADER_SIZE..BASE_HEADER_SIZE + payload.len()].copy_from_slice(payload);
    verifying_key
        .verify_strict(&message[..BASE_HEADER_SIZE + payload.len()], &signature)
        .is_ok()
}
