from __future__ import annotations

import io

from src.core import log_capture


def test_component_log_store_append_filter_clear() -> None:
    store = log_capture.ComponentLogStore(max_entries_per_node=2)
    store.register_thread("node-a")
    store.append_from_thread("stdout", "hello")
    store.append_from_thread("stdout", " world\n")
    store.append_from_thread("stderr", "err\n")
    entries = store.get_entries("node-a")
    assert len(entries) == 2
    assert entries[-1].stream == "stderr"
    assert len(store.get_entries("node-a", limit=1)) == 1
    assert len(store.get_entries("node-a", after=entries[0].seq, limit=1)) == 1
    store.clear_node("node-a")
    assert store.get_entries("node-a") == []
    assert isinstance(log_capture.get_log_store(), log_capture.ComponentLogStore)


def test_unregister_flushes_partial_and_handles_missing_thread() -> None:
    store = log_capture.ComponentLogStore()
    store.register_thread("node-y")
    store.unregister_thread()
    store.register_thread("node-x", ident=123)
    store._partials[(123, "stdout")] = "partial"
    store.unregister_thread(ident=123)
    assert store.get_entries("node-x")[-1].text == "partial"
    store.unregister_thread(ident=999)


def test_routed_stream_and_install_global_capture(monkeypatch) -> None:
    fake_out = io.StringIO()
    fake_err = io.StringIO()
    monkeypatch.setattr(log_capture.sys, "stdout", fake_out)
    monkeypatch.setattr(log_capture.sys, "stderr", fake_err)
    monkeypatch.setattr(log_capture, "_store", log_capture.ComponentLogStore())
    monkeypatch.setattr(log_capture, "_installed", False)

    log_capture.install_global_stream_capture()
    assert log_capture._installed is True
    routed = log_capture.sys.stdout
    assert routed.isatty() is False
    assert routed.write("") == 0
    assert routed.write("x") == 1
    routed.flush()
    log_capture.install_global_stream_capture()
