//! Content identity: stable for the same bytes, different for changed bytes.

use vault::AssetHash;

/// The low half is xxHash64 with seed 0, so the published vectors pin the
/// algorithm. Without this the hash could be stable and wrong at the same time.
fn low(hash: AssetHash) -> u64 {
    let bytes = hash.to_bytes();
    u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"))
}

#[test]
fn the_hash_matches_the_xxhash64_reference_vectors() {
    assert_eq!(low(AssetHash::of(b"")), 0xEF46_DB37_51D8_E999);
    assert_eq!(low(AssetHash::of(b"a")), 0xD24E_C4F1_A98C_6E5B);
    // 40 bytes takes the stripe loop, which the short vectors never reach.
    assert_eq!(low(AssetHash::of(&[b'x'; 40])), 0x926F_564E_1B3E_18D5);
}

#[test]
fn the_same_bytes_always_hash_the_same() {
    let first = AssetHash::of(b"texture bytes");
    let second = AssetHash::of(b"texture bytes");
    assert_eq!(first, second);
    assert_eq!(first.to_hex(), second.to_hex());
}

#[test]
fn a_single_changed_byte_is_a_different_identity() {
    let before = AssetHash::of(b"frame 0001");
    let after = AssetHash::of(b"frame 0002");
    assert_ne!(before, after);
    // A length-preserving change is the case a size check alone would miss.
    assert_eq!(before.to_hex().len(), after.to_hex().len());
}

#[test]
fn bytes_round_trip_and_hex_is_the_address() {
    let hash = AssetHash::of(b"round trip");
    assert_eq!(AssetHash::from_bytes(hash.to_bytes()), hash);
    assert_eq!(hash.to_hex(), hash.to_string());
    assert_eq!(hash.to_hex().len(), 32);
}
