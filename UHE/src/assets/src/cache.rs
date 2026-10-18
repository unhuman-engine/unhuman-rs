use std::collections::HashMap;

use crate::cache_hash::AssetHash;

/// Decoded entries a cache holds before it starts evicting. A number rather than
/// unbounded, because the point of a cache is to trade a bounded amount of
/// memory for the decodes it saves.
pub const DEFAULT_CAPACITY: usize = 256;

/// A bounded, content-addressed cache.
///
/// The shape is `Font2D::s_Cache`: [`Cache::get_or_insert_with`] looks the key
/// up and, on a miss, builds the value once and stores it, so a second request
/// for the same content never pays for the decode. It adds the two things a
/// font cache did not need and a long-lived registry does: a capacity and
/// least-recently-used eviction when the capacity is reached.
///
/// Recency is a monotonic counter per entry. A lookup is one increment; an
/// eviction scans for the smallest counter. The scan is linear in the capacity,
/// which is small next to the decode it saves, and it keeps the hit path free of
/// allocation and pointer chasing.
#[derive(Debug)]
pub struct Cache<V> {
    entries: HashMap<AssetHash, Entry<V>>,
    capacity: usize,
    clock: u64,
    evictions: u64,
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    used: u64,
}

impl<V> Cache<V> {
    /// A cache that holds at most `capacity` entries. Zero is clamped to one so
    /// an insert can always make room for the value it was asked to store.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: capacity.max(1),
            clock: 0,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many entries have been dropped to respect the capacity.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn contains(&self, key: AssetHash) -> bool {
        self.entries.contains_key(&key)
    }

    /// The entry at `key`, marked most recently used.
    pub fn get(&mut self, key: AssetHash) -> Option<&V> {
        self.clock = self.clock.wrapping_add(1);
        let used = self.clock;
        let entry = self.entries.get_mut(&key)?;
        entry.used = used;
        Some(&entry.value)
    }

    /// The entry at `key`, leaving its recency alone.
    pub fn peek(&self, key: AssetHash) -> Option<&V> {
        self.entries.get(&key).map(|entry| &entry.value)
    }

    /// Stores `value` at `key`. A value already there is replaced and returned;
    /// otherwise the least recently used entry is evicted once the cache is
    /// full, and `None` is returned.
    pub fn insert(&mut self, key: AssetHash, value: V) -> Option<V> {
        self.clock = self.clock.wrapping_add(1);
        let used = self.clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.used = used;
            return Some(std::mem::replace(&mut entry.value, value));
        }
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }
        self.entries.insert(key, Entry { value, used });
        None
    }

    /// The value at `key`, building and storing it on a miss: the one call that
    /// is exactly `Font2D::Get`.
    pub fn get_or_insert_with(&mut self, key: AssetHash, build: impl FnOnce() -> V) -> &V {
        if !self.entries.contains_key(&key) {
            let value = build();
            let _ = self.insert(key, value);
        }
        self.get(key).expect("present after the insert above")
    }

    pub fn remove(&mut self, key: AssetHash) -> Option<V> {
        self.entries.remove(&key).map(|entry| entry.value)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn evict_lru(&mut self) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.used)
            .map(|(key, _)| *key);
        if let Some(key) = oldest {
            self.entries.remove(&key);
            self.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(n: u8) -> AssetHash {
        AssetHash::of(&[n])
    }

    #[test]
    fn a_hit_returns_the_value_and_a_miss_returns_nothing() {
        let mut cache = Cache::new(4);
        assert_eq!(cache.get(hash(1)), None);
        assert!(cache.is_empty());

        cache.insert(hash(1), "one");
        assert_eq!(cache.get(hash(1)), Some(&"one"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.evictions(), 0);
    }

    #[test]
    fn inserting_past_capacity_evicts_the_least_recently_used() {
        let mut cache = Cache::new(2);
        cache.insert(hash(1), "one");
        cache.insert(hash(2), "two");

        // Touching one must save it from eviction, so two is the victim.
        assert_eq!(cache.get(hash(1)), Some(&"one"));
        cache.insert(hash(3), "three");

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.evictions(), 1);
        assert!(cache.contains(hash(1)));
        assert!(!cache.contains(hash(2)));
        assert!(cache.contains(hash(3)));
    }

    #[test]
    fn replacing_a_key_returns_the_old_value_and_does_not_evict() {
        let mut cache = Cache::new(2);
        cache.insert(hash(1), "old");
        cache.insert(hash(2), "two");

        assert_eq!(cache.insert(hash(1), "new"), Some("old"));
        assert_eq!(cache.evictions(), 0);
        assert_eq!(cache.get(hash(1)), Some(&"new"));
        assert!(cache.contains(hash(2)));
    }

    #[test]
    fn get_or_insert_builds_once_and_then_hits() {
        let mut cache = Cache::new(2);
        let mut built = 0;

        for _ in 0..3 {
            let value = cache.get_or_insert_with(hash(1), || {
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
        cache.insert(hash(1), 1);
        cache.insert(hash(2), 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.evictions(), 1);
    }
}
