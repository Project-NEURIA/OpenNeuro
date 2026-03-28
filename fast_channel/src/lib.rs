use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const MAX_CURSORS: usize = 16;

/// A slot in the ring buffer. Each slot has its own lock to minimize contention.
/// The PyObject is behind a Mutex because we need GIL-safe refcount operations.
struct Slot {
    item: Mutex<Option<Py<PyAny>>>,
}

/// Ring buffer with CAS-based sequencing.
///
/// Hot path (send/receive): atomic load/store/CAS on sequence counters.
/// Cold path (register/unregister): mutex on cursor array.
///
/// Layout:
///   - buffer: fixed-size array of slots
///   - write_seq: atomic counter, incremented via fetch_add on each send
///   - cursors: fixed-size array of atomic counters (one per subscriber)
///   - cursor_active: which cursor slots are in use (protected by reg_lock)
struct RingBuffer {
    buffer: Vec<Slot>,
    capacity: usize,
    write_seq: AtomicUsize,

    // Per-cursor atomic positions — CAS on the hot path
    cursor_pos: Vec<AtomicUsize>,
    cursor_sub_ids: Mutex<Vec<Option<u64>>>, // maps slot index -> sub_id

    // Only used for blocking wait
    notify: Condvar,
    wait_lock: Mutex<()>,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(Slot {
                item: Mutex::new(None),
            });
        }

        let mut cursor_pos = Vec::with_capacity(MAX_CURSORS);
        for _ in 0..MAX_CURSORS {
            cursor_pos.push(AtomicUsize::new(0));
        }

        let mut sub_ids = Vec::with_capacity(MAX_CURSORS);
        sub_ids.resize_with(MAX_CURSORS, || None);

        Self {
            buffer,
            capacity,
            write_seq: AtomicUsize::new(0),
            cursor_pos,
            cursor_sub_ids: Mutex::new(sub_ids),
            notify: Condvar::new(),
            wait_lock: Mutex::new(()),
        }
    }

    #[inline]
    fn slot_index(&self, seq: usize) -> usize {
        seq & (self.capacity - 1) // capacity must be power of 2
    }

    /// Find the cursor index for a sub_id. Returns None if not registered.
    fn cursor_index(&self, sub_id: u64) -> Option<usize> {
        let ids = self.cursor_sub_ids.lock();
        ids.iter().position(|id| *id == Some(sub_id))
    }
}

fn next_power_of_2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

#[pyclass]
#[derive(Clone)]
struct RustChannel {
    ring: Arc<RingBuffer>,
}

#[pymethods]
impl RustChannel {
    #[new]
    #[pyo3(signature = (capacity=65536))]
    fn new(capacity: usize) -> Self {
        let capacity = next_power_of_2(capacity.max(16));
        Self {
            ring: Arc::new(RingBuffer::new(capacity)),
        }
    }

    /// Send: write item to slot, advance write_seq via atomic fetch_add, notify.
    fn send(&self, item: Py<PyAny>) {
        let ring = &self.ring;
        // Claim a slot atomically (CAS via fetch_add)
        let seq = ring.write_seq.fetch_add(1, Ordering::Release);
        let idx = ring.slot_index(seq);

        // Write the item (per-slot lock, very brief)
        {
            let mut slot = ring.buffer[idx].item.lock();
            *slot = Some(item);
        }

        // Wake waiting consumers
        ring.notify.notify_all();
    }

    /// Register a subscriber cursor.
    fn register(&self, sub_id: u64, latest: bool) {
        let ring = &self.ring;
        let mut ids = ring.cursor_sub_ids.lock();

        // Find a free slot
        let slot_idx = ids.iter().position(|id| id.is_none())
            .expect("Too many cursors (max 16)");

        let pos = if latest {
            ring.write_seq.load(Ordering::Acquire)
        } else {
            let ws = ring.write_seq.load(Ordering::Acquire);
            ws.saturating_sub(ring.capacity)
        };

        ring.cursor_pos[slot_idx].store(pos, Ordering::Release);
        ids[slot_idx] = Some(sub_id);
    }

    /// Unregister a subscriber.
    fn unregister(&self, sub_id: u64) {
        let ring = &self.ring;
        let mut ids = ring.cursor_sub_ids.lock();
        if let Some(idx) = ids.iter().position(|id| *id == Some(sub_id)) {
            ids[idx] = None;
        }
    }

    /// Blocking receive using CAS to advance cursor.
    fn wait_and_get(&self, py: Python<'_>, sub_id: u64, stop_fn: Py<PyAny>) -> PyResult<Option<PyObject>> {
        let ring = &self.ring;
        let cursor_idx = match ring.cursor_index(sub_id) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        loop {
            let cursor = ring.cursor_pos[cursor_idx].load(Ordering::Acquire);
            let write = ring.write_seq.load(Ordering::Acquire);

            if cursor < write {
                // Data available — CAS to claim this position
                match ring.cursor_pos[cursor_idx].compare_exchange(
                    cursor,
                    cursor + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // We claimed it. Read the item.
                        let idx = ring.slot_index(cursor);
                        let slot = ring.buffer[idx].item.lock();
                        let item = slot.as_ref().map(|obj| obj.clone_ref(py));
                        return Ok(item.map(|o| o.into()));
                    }
                    Err(_) => {
                        // Another thread advanced our cursor (shouldn't happen in
                        // single-consumer-per-cursor model, but retry to be safe)
                        continue;
                    }
                }
            }

            // Nothing available — wait with GIL released
            let ring_ref = ring.clone();
            py.allow_threads(move || {
                let mut guard = ring_ref.wait_lock.lock();
                ring_ref.notify.wait_for(&mut guard, Duration::from_millis(100));
            });

            // Check stop
            let should_stop: bool = stop_fn.call0(py)?.extract(py)?;
            if should_stop {
                return Ok(None);
            }
        }
    }

    /// Non-blocking receive using CAS.
    fn try_get(&self, py: Python<'_>, sub_id: u64) -> Option<PyObject> {
        let ring = &self.ring;
        let cursor_idx = ring.cursor_index(sub_id)?;

        let cursor = ring.cursor_pos[cursor_idx].load(Ordering::Acquire);
        let write = ring.write_seq.load(Ordering::Acquire);

        if cursor >= write {
            return None;
        }

        // CAS to advance cursor
        match ring.cursor_pos[cursor_idx].compare_exchange(
            cursor,
            cursor + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                let idx = ring.slot_index(cursor);
                let slot = ring.buffer[idx].item.lock();
                slot.as_ref().map(|obj| obj.clone_ref(py).into())
            }
            Err(_) => None, // Contended, caller can retry
        }
    }

    /// Skip to the latest item.
    fn fast_forward(&self, sub_id: u64) {
        let ring = &self.ring;
        if let Some(cursor_idx) = ring.cursor_index(sub_id) {
            let write = ring.write_seq.load(Ordering::Acquire);
            if write > 0 {
                // Atomically set cursor to write - 1
                ring.cursor_pos[cursor_idx].store(write - 1, Ordering::Release);
            }
        }
    }

    /// How many items behind.
    fn lag(&self, sub_id: u64) -> usize {
        let ring = &self.ring;
        match ring.cursor_index(sub_id) {
            Some(idx) => {
                let write = ring.write_seq.load(Ordering::Acquire);
                let cursor = ring.cursor_pos[idx].load(Ordering::Acquire);
                write.saturating_sub(cursor)
            }
            None => 0,
        }
    }

    fn write_pos(&self) -> usize {
        self.ring.write_seq.load(Ordering::Acquire)
    }
}

#[pymodule]
fn _fast_channel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustChannel>()?;
    Ok(())
}
