//! Content addressing: the stable identity of a byte string.
//!
//! Split from `id.rs` because it is a self-contained algorithm with a public
//! type, not part of the runtime [`AssetId`](crate::AssetId) scheme, and because
//! the reference implementation is long enough to deserve its own file.

use std::sync::Arc;

/// The seed for the second 64-bit half, so the two halves are independent
/// digests rather than the same one written twice.
const SEED_HIGH: u64 = 0x9E37_79B9_7F4A_7C15;

/// A stable content address for a byte string: the same bytes always hash the
/// same and a change of any size hashes differently.
///
/// This is the identity a registry needs and a filesystem does not give.
/// Modification time cannot serve: a copy or a checkout does not preserve it,
/// its resolution is coarse enough that two writes inside one tick are
/// indistinguishable, and `touch` moves it without touching content. A hash of
/// the bytes is the only value that follows the content to whichever machine or
/// archive holds it.
///
/// Two seeded 64-bit xxHash passes make a 128-bit address. It is a fingerprint,
/// not a cryptographic digest: it catches an accidental change, not a forged
/// collision.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetHash {
    low: u64,
    high: u64,
}

impl AssetHash {
    /// Hashes `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            low: xxh64(bytes, 0),
            high: xxh64(bytes, SEED_HIGH),
        }
    }

    /// The two halves as little-endian bytes, for a filename or a manifest key.
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.low.to_le_bytes());
        out[8..].copy_from_slice(&self.high.to_le_bytes());
        out
    }

    /// The inverse of [`AssetHash::to_bytes`].
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut low = [0u8; 8];
        let mut high = [0u8; 8];
        low.copy_from_slice(&bytes[..8]);
        high.copy_from_slice(&bytes[8..]);
        Self {
            low: u64::from_le_bytes(low),
            high: u64::from_le_bytes(high),
        }
    }

    /// Lower-case hex, high half first, the way a digest is usually written.
    pub fn to_hex(self) -> String {
        format!("{:016x}{:016x}", self.high, self.low)
    }
}

impl std::fmt::Debug for AssetHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AssetHash({self})")
    }
}

impl std::fmt::Display for AssetHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}{:016x}", self.high, self.low)
    }
}

/// A cache identity: a hash for the bucket and the full content for the key.
///
/// [`AssetHash`] only picks a bucket. Two different byte strings can share one,
/// so an entry is a hit only when the content also matches; otherwise both
/// live in the same bucket and neither aliases the other.
#[derive(Clone, Debug)]
pub struct ContentKey {
    hash: AssetHash,
    content: Arc<[u8]>,
    /// What else the decode depended on, such as the loader kind or a model's
    /// directory. Compared along with `content`.
    context: Arc<[u8]>,
}

impl ContentKey {
    /// Builds a key from a bucket hash, the full bytes and their context.
    pub fn new(
        hash: AssetHash,
        content: impl Into<Arc<[u8]>>,
        context: impl Into<Arc<[u8]>>,
    ) -> Self {
        Self {
            hash,
            content: content.into(),
            context: context.into(),
        }
    }

    /// The bucket this key belongs to.
    pub fn hash(&self) -> AssetHash {
        self.hash
    }
}

impl PartialEq for ContentKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && (Arc::ptr_eq(&self.content, &other.content) || self.content == other.content)
            && (Arc::ptr_eq(&self.context, &other.context) || self.context == other.context)
    }
}

impl Eq for ContentKey {}

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

/// xxHash64, the reference algorithm.
///
/// Hand-rolled and stable across runs, platforms and compilers because it is
/// pure integer arithmetic. `DefaultHasher` is neither: Rust does not promise
/// its output survives a release, which an on-disk identity cannot accept.
fn xxh64(input: &[u8], seed: u64) -> u64 {
    let mut i = 0;
    let mut h;

    if input.len() >= 32 {
        let (mut v1, mut v2, mut v3, mut v4) = (
            seed.wrapping_add(P1).wrapping_add(P2),
            seed.wrapping_add(P2),
            seed,
            seed.wrapping_sub(P1),
        );
        while i + 32 <= input.len() {
            v1 = round(v1, word(input, i));
            v2 = round(v2, word(input, i + 8));
            v3 = round(v3, word(input, i + 16));
            v4 = round(v4, word(input, i + 24));
            i += 32;
        }
        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = seed.wrapping_add(P5);
    }

    h = h.wrapping_add(input.len() as u64);
    while i + 8 <= input.len() {
        h ^= round(0, word(input, i));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= input.len() {
        h ^= (dword(input, i) as u64).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    while i < input.len() {
        h ^= (input[i] as u64).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
        i += 1;
    }

    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}

fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

fn merge(acc: u64, lane: u64) -> u64 {
    (acc ^ round(0, lane)).wrapping_mul(P1).wrapping_add(P4)
}

fn word(input: &[u8], at: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[at..at + 8]);
    u64::from_le_bytes(bytes)
}

fn dword(input: &[u8], at: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&input[at..at + 4]);
    u32::from_le_bytes(bytes)
}
