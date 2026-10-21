use std::collections::HashMap;

use crate::cache_hash::{AssetHash, ContentKey};

/// Decoded entries a cache holds before it starts evicting. A number rather than
/// unbounded, because the point of a cache is to trade a bounded amount of
/// memory for the decodes it saves.
pub const DEFAULT_CAPACITY: usize = 256;

/// A bounded, content-addressed cache.
///
/// The policy is explicit: at most `capacity` entries, an optional byte budget,
/// and least-recently-used eviction on either limit. `ContentKey` carries the
/// full bytes as well as the hash, so a hash collision is a second entry in the
/// same bucket, not an alias.
///
/// Recency is a monotonic counter per entry. A lookup is one increment; an
/// eviction scans for the smallest counter. The scan is linear in the entry
/// count, which is small next to the decode it saves, and it keeps the hit path
/// free of allocation and pointer chasing.
#[derive(Debug)]
pub struct Cache<V> {
    /// Several entries may share a hash, so a bucket is a list rather than one
    /// slot.
    buckets: HashMap<AssetHash, Vec<Entry<V>>>,
    capacity: usize,
    /// `None` means no byte limit. A value of zero also disables the limit.
    byte_capacity: Option<usize>,
    bytes: usize,
    entries: usize,
    clock: u64,
    evictions: u64,
}

#[derive(Debug)]
struct Entry<V> {
    key: ContentKey,
    value: V,
    /// Memory charged against `byte_capacity`, supplied by the caller because
    /// the cache cannot see inside an `Arc`.
    size: usize,
    used: u64,
}

impl<V> Cache<V> {
    /// A cache that holds at most `capacity` entries and has no byte limit.
    /// Zero is clamped to one so an insert can always make room for the value
    /// it was asked to store.
    pub fn new(capacity: usize) -> Self {
        Self::with_byte_capacity(capacity, None)
    }

    /// A cache bounded by entries and, when `byte_capacity` is positive, by the
    /// sum of the sizes passed to [`Cache::insert_sized`].
    pub fn with_byte_capacity(capacity: usize, byte_capacity: Option<usize>) -> Self {
        Self {
            buckets: HashMap::new(),
            capacity: capacity.max(1),
            byte_capacity: byte_capacity.filter(|&bytes| bytes > 0),
            bytes: 0,
            entries: 0,
            clock: 0,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn byte_capacity(&self) -> Option<usize> {
        self.byte_capacity
    }

    /// The total size charged by the entries currently held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// How many entries have been dropped to respect a limit.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn contains(&self, key: &ContentKey) -> bool {
        self.locate(key).is_some()
    }

    /// The entry at `key`, marked most recently used.
    pub fn get(&mut self, key: &ContentKey) -> Option<&V> {
        let used = self.tick();
        let (hash, index) = self.locate(key)?;
        let entry = &mut self.buckets.get_mut(&hash)?[index];
        entry.used = used;
        Some(&entry.value)
    }

    /// The entry at `key`, leaving its recency alone.
    pub fn peek(&self, key: &ContentKey) -> Option<&V> {
        let (hash, index) = self.locate(key)?;
        self.buckets.get(&hash).map(|bucket| &bucket[index].value)
    }

    /// Stores `value` at `key`, charging `size_of::<V>()` bytes. Callers whose
    /// value owns memory outside itself should use [`Cache::insert_sized`].
    pub fn insert(&mut self, key: ContentKey, value: V) -> Option<V> {
        self.insert_sized(key, std::mem::size_of::<V>(), value)
    }

    /// Stores `value` at `key`, charging `size` against the byte limit.
    ///
    /// A value already there is replaced and returned; otherwise the least
    /// recently used entry is evicted until both limits fit, and `None` is
    /// returned. A value larger than the whole byte budget is not cached.
    pub fn insert_sized(&mut self, key: ContentKey, size: usize, value: V) -> Option<V> {
        let used = self.tick();

        if let Some((hash, index)) = self.locate(&key) {
            let old_size = self.buckets[&hash][index].size;
            if self.byte_capacity.is_some_and(|capacity| size > capacity) {
                return Some(self.remove_entry(hash, index, false).value);
            }
            while self
                .byte_capacity
                .is_some_and(|capacity| self.bytes - old_size + size > capacity)
            {
                if !self.evict_lru_except(Some(&key)) {
                    break;
                }
            }
            if self
                .byte_capacity
                .is_some_and(|capacity| self.bytes - old_size + size > capacity)
            {
                return Some(self.remove_entry(hash, index, false).value);
            }

            // Evicting another colliding entry can shift this one's index, so
            // locate it again rather than trusting the index from before.
            let (hash, index) = self.locate(&key).expect("excluded from eviction");
            self.bytes = self.bytes - old_size + size;
            let entry = &mut self.buckets.get_mut(&hash).expect("located above")[index];
            entry.key = key;
            entry.size = size;
            entry.used = used;
            return Some(std::mem::replace(&mut entry.value, value));
        }

        if self.byte_capacity.is_some_and(|capacity| size > capacity) {
            return None;
        }
        while self.entries >= self.capacity {
            if !self.evict_lru_except(None) {
                break;
            }
        }
        while self
            .byte_capacity
            .is_some_and(|capacity| self.bytes + size > capacity)
        {
            if !self.evict_lru_except(None) {
                break;
            }
        }
        if self
            .byte_capacity
            .is_some_and(|capacity| self.bytes + size > capacity)
        {
            return None;
        }

        self.buckets.entry(key.hash()).or_default().push(Entry {
            key,
            value,
            size,
            used,
        });
        self.entries += 1;
        self.bytes += size;
        None
    }

    /// The value at `key`, building and storing it on a miss. The size charged
    /// is `size_of::<V>()`; use [`Cache::insert_sized`] when the value owns
    /// memory.
    pub fn get_or_insert_with(&mut self, key: ContentKey, build: impl FnOnce() -> V) -> &V {
        if !self.contains(&key) {
            let value = build();
            let _ = self.insert(key.clone(), value);
        }
        self.get(&key).expect("present after the insert above")
    }

    pub fn remove(&mut self, key: &ContentKey) -> Option<V> {
        let (hash, index) = self.locate(key)?;
        Some(self.remove_entry(hash, index, false).value)
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.entries = 0;
        self.bytes = 0;
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn locate(&self, key: &ContentKey) -> Option<(AssetHash, usize)> {
        let bucket = self.buckets.get(&key.hash())?;
        let index = bucket.iter().position(|entry| entry.key == *key)?;
        Some((key.hash(), index))
    }

    /// Drops the least recently used entry that is not `exclude`, returning
    /// whether there was one.
    fn evict_lru_except(&mut self, exclude: Option<&ContentKey>) -> bool {
        let victim = self
            .buckets
            .iter()
            .flat_map(|(hash, bucket)| bucket.iter().enumerate().map(move |(i, e)| (*hash, i, e)))
            .filter(|(_, _, entry)| exclude.is_none_or(|key| entry.key != *key))
            .min_by_key(|(_, _, entry)| entry.used)
            .map(|(hash, index, _)| (hash, index));

        let Some((hash, index)) = victim else {
            return false;
        };
        self.remove_entry(hash, index, true);
        true
    }

    fn remove_entry(&mut self, hash: AssetHash, index: usize, eviction: bool) -> Entry<V> {
        let bucket = self.buckets.get_mut(&hash).expect("bucket exists");
        let entry = bucket.remove(index);
        if bucket.is_empty() {
            self.buckets.remove(&hash);
        }
        self.entries -= 1;
        self.bytes = self.bytes.saturating_sub(entry.size);
        if eviction {
            self.evictions += 1;
        }
        entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> ContentKey {
        let content = [n];
        ContentKey::new(AssetHash::of(&content), content.to_vec(), Vec::new())
    }

    fn colliding(hash: AssetHash, content: &[u8]) -> ContentKey {
        ContentKey::new(hash, content.to_vec(), Vec::new())
    }

    #[test]
    fn a_hit_returns_the_value_and_a_miss_returns_nothing() {
        let mut cache = Cache::new(4);
        assert_eq!(cache.get(&key(1)), None);
        assert!(cache.is_empty());

        cache.insert(key(1), "one");
        assert_eq!(cache.get(&key(1)), Some(&"one"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.evictions(), 0);
    }

    #[test]
    fn equal_hashes_with_different_content_are_different_entries() {
        let hash = AssetHash::of(b"bucket");
        let first = colliding(hash, b"first");
        let second = colliding(hash, b"second");

        let mut cache = Cache::new(4);
        cache.insert(first.clone(), "one");
        cache.insert(second.clone(), "two");

        assert_eq!(cache.len(), 2, "a collision replaced an unrelated entry");
        assert_eq!(cache.get(&first), Some(&"one"));
        assert_eq!(cache.get(&second), Some(&"two"));
    }

    #[test]
    fn inserting_past_capacity_evicts_the_least_recently_used() {
        let mut cache = Cache::new(2);
        cache.insert(key(1), "one");
        cache.insert(key(2), "two");

        // Touching one must save it from eviction, so two is the victim.
        assert_eq!(cache.get(&key(1)), Some(&"one"));
        cache.insert(key(3), "three");

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.evictions(), 1);
        assert!(cache.contains(&key(1)));
        assert!(!cache.contains(&key(2)));
        assert!(cache.contains(&key(3)));
    }

    #[test]
    fn a_byte_limit_evicts_the_oldest_until_the_new_entry_fits() {
        let mut cache = Cache::with_byte_capacity(10, Some(10));
        cache.insert_sized(key(1), 2, "one");
        cache.insert_sized(key(2), 2, "two");
        cache.get(&key(1));
        cache.insert_sized(key(3), 2, "three");

        // The 8-byte entry needs room: two is oldest, then one, leaving three.
        cache.insert_sized(key(4), 8, "four");

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 10);
        assert_eq!(cache.evictions(), 2);
        assert!(cache.contains(&key(3)));
        assert!(cache.contains(&key(4)));
        assert!(!cache.contains(&key(1)));
        assert!(!cache.contains(&key(2)));
    }

    #[test]
    fn an_entry_larger_than_the_byte_budget_is_not_cached() {
        let mut cache = Cache::with_byte_capacity(4, Some(8));
        assert_eq!(cache.insert_sized(key(1), 9, "too large"), None);
        assert!(cache.is_empty());
    }

    #[test]
    fn replacing_a_colliding_entry_survives_its_bucket_shifting() {
        let hash = AssetHash::of(b"bucket");
        let (a, b, c) = (
            colliding(hash, b"a"),
            colliding(hash, b"b"),
            colliding(hash, b"c"),
        );
        let mut cache = Cache::with_byte_capacity(4, Some(3));
        cache.insert_sized(a.clone(), 1, "a");
        cache.insert_sized(b.clone(), 1, "b");
        cache.insert_sized(c.clone(), 1, "c");

        // Growing c evicts a, which is ahead of c in the same bucket.
        assert_eq!(cache.insert_sized(c.clone(), 2, "C"), Some("c"));

        assert!(!cache.contains(&a));
        assert!(cache.contains(&b));
        assert_eq!(cache.get(&c), Some(&"C"));
    }

    #[test]
    fn replacing_a_key_returns_the_old_value_and_does_not_evict() {
        let mut cache = Cache::new(2);
        cache.insert(key(1), "old");
        cache.insert(key(2), "two");

        assert_eq!(cache.insert(key(1), "new"), Some("old"));
        assert_eq!(cache.evictions(), 0);
        assert_eq!(cache.get(&key(1)), Some(&"new"));
        assert!(cache.contains(&key(2)));
    }

    #[test]
    fn get_or_insert_builds_once_and_then_hits() {
        let mut cache = Cache::new(2);
        let mut built = 0;

        for _ in 0..3 {
            let value = cache.get_or_insert_with(key(1), || {
                built += 1;
                "value"
            });
            assert_eq!(value, &"value");
        }

        assert_eq!(built, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_zero_capacity_is_clamped_so_an_insert_has_somewhere_to_go() {
        let mut cache = Cache::new(0);
        assert_eq!(cache.capacity(), 1);
        cache.insert(key(1), 1);
        cache.insert(key(2), 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.evictions(), 1);
    }
}
