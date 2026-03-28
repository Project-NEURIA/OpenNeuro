use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Shared ring buffer state protected by a single Mutex + Condvar.
/// No per-message GC — old slots are simply overwritten when the ring wraps.
struct Inner {
    buffer: Vec<Option<Py<PyAny>>>,
    capacity: usize,
    write_pos: usize, // monotonically increasing sequence number
    cursors: HashMap<u64, usize>, // sub_id -> read position (sequence number)
}

impl Inner {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        Self {
            buffer,
            capacity,
            write_pos: 0,
            cursors: HashMap::new(),
        }
    }

    #[inline]
    fn slot(&self, seq: usize) -> usize {
        seq % self.capacity
    }
}

/// The core channel: a fixed-size ring buffer with multi-cursor support.
#[pyclass]
#[derive(Clone)]
struct RustChannel {
    inner: Arc<Mutex<Inner>>,
    condvar: Arc<Condvar>,
}

#[pymethods]
impl RustChannel {
    #[new]
    #[pyo3(signature = (capacity=65536))]
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(capacity))),
            condvar: Arc::new(Condvar::new()),
        }
    }

    /// Append an item and notify all waiting consumers.
    fn send(&self, item: Py<PyAny>) {
        let mut inner = self.inner.lock();
        let slot = inner.slot(inner.write_pos);
        inner.buffer[slot] = Some(item);
        inner.write_pos += 1;
        self.condvar.notify_all();
    }

    /// Register a subscriber cursor.
    fn register(&self, sub_id: u64, latest: bool) {
        let mut inner = self.inner.lock();
        let pos = if latest {
            inner.write_pos
        } else {
            inner.write_pos.saturating_sub(inner.capacity)
        };
        inner.cursors.insert(sub_id, pos);
    }

    /// Unregister a subscriber.
    fn unregister(&self, sub_id: u64) {
        let mut inner = self.inner.lock();
        inner.cursors.remove(&sub_id);
    }

    /// Blocking receive. Returns None if stop_fn() returns True.
    fn wait_and_get(&self, py: Python<'_>, sub_id: u64, stop_fn: Py<PyAny>) -> PyResult<Option<PyObject>> {
        loop {
            // Try to get an item under the lock
            {
                let mut inner = self.inner.lock();
                let cursor = match inner.cursors.get(&sub_id) {
                    Some(&c) => c,
                    None => return Ok(None),
                };

                if cursor < inner.write_pos {
                    let slot = inner.slot(cursor);
                    let item = inner.buffer[slot].as_ref().map(|obj| obj.clone_ref(py));
                    inner.cursors.insert(sub_id, cursor + 1);
                    return Ok(item.map(|o| o.into()));
                }
            } // lock dropped here

            // Wait with GIL released so other Python threads can send
            let inner_ref = self.inner.clone();
            let condvar_ref = self.condvar.clone();
            py.allow_threads(move || {
                let mut inner = inner_ref.lock();
                condvar_ref.wait_for(&mut inner, Duration::from_millis(100));
            });

            // Check stop condition (GIL held)
            let should_stop: bool = stop_fn.call0(py)?.extract(py)?;
            if should_stop {
                return Ok(None);
            }
        }
    }

    /// Non-blocking receive.
    fn try_get(&self, py: Python<'_>, sub_id: u64) -> Option<PyObject> {
        let mut inner = self.inner.lock();
        let cursor = *inner.cursors.get(&sub_id)?;
        if cursor >= inner.write_pos {
            return None;
        }
        let slot = inner.slot(cursor);
        let item = inner.buffer[slot].as_ref().map(|obj| obj.clone_ref(py));
        inner.cursors.insert(sub_id, cursor + 1);
        item.map(|o| o.into())
    }

    /// Skip to the latest item.
    fn fast_forward(&self, sub_id: u64) {
        let mut inner = self.inner.lock();
        let write_pos = inner.write_pos;
        if let Some(cursor) = inner.cursors.get_mut(&sub_id) {
            if write_pos > *cursor + 1 {
                *cursor = write_pos - 1;
            }
        }
    }

    /// How many items behind the consumer is.
    fn lag(&self, sub_id: u64) -> usize {
        let inner = self.inner.lock();
        match inner.cursors.get(&sub_id) {
            Some(&cursor) => inner.write_pos.saturating_sub(cursor),
            None => 0,
        }
    }

    /// Current write position.
    fn write_pos(&self) -> usize {
        self.inner.lock().write_pos
    }
}

#[pymodule]
fn _fast_channel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustChannel>()?;
    Ok(())
}
