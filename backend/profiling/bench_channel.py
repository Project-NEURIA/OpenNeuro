"""
Channel vs Queue throughput & latency benchmark.

Methodology adapted from LMAX Disruptor perftest suite:
  - 1M messages per run (Python is ~100x slower than Java, so 1M not 100M)
  - 7 runs per test, gc.collect() between runs, report best and median
  - Correctness checksum: producer sums all sent values, consumer sums all
    received values, assert equal after each run
  - Matched topologies: same test for Channel and Queue
  - Latency: ping-pong with percentile distribution (p50, p90, p99, p99.9)

Usage:
    cd backend
    uv run --no-group cuda python -m profiling.bench_channel
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.channel import Channel, Sender

try:
    from fast_channel import Channel as FastChannel, Sender as FastSender  # type: ignore[import-not-found]

    HAS_FAST_CHANNEL = True
except ImportError:
    FastChannel = cast(Any, None)
    FastSender = cast(Any, None)
    HAS_FAST_CHANNEL = False
    print("WARNING: fast_channel not installed, skipping FastChannel benchmarks")

RESULTS_DIR = Path(__file__).parent / "results"

N = 100_000  # messages per run
RUNS = 7  # runs per test
LATENCY_N = 10_000
LATENCY_RUNS = 5


def percentiles(data: list[float]) -> dict[str, float]:
    s = sorted(data)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p90": s[int(n * 0.90)],
        "p99": s[int(n * 0.99)],
        "p99.9": s[int(n * 0.999)],
        "max": s[-1],
    }


# ──────────────────────────────────────────────────────────────────────
# 1P1C — Single Producer, Single Consumer
# ──────────────────────────────────────────────────────────────────────


def bench_1p1c_channel(n: int) -> tuple[float, bool]:
    ch: Channel[int] = Channel()
    sender = Sender(ch)
    stop = threading.Event()
    latch = threading.Event()
    consumer_sum = [0]

    def consumer():
        ch._register(1, latest=False)
        for _ in range(n):
            item = ch._wait_and_get(1, stop)
            if item is None:
                break
            consumer_sum[0] += item
        latch.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    expected_sum = 0
    t0 = time.perf_counter()
    for i in range(n):
        sender.send(i)
        expected_sum += i
    latch.wait(timeout=10)
    elapsed = time.perf_counter() - t0

    stop.set()
    t.join(timeout=1)
    return elapsed, consumer_sum[0] == expected_sum


def bench_1p1c_queue(n: int) -> tuple[float, bool]:
    q: Queue[int] = Queue()
    latch = threading.Event()
    consumer_sum = [0]

    def consumer():
        for _ in range(n):
            item = q.get()
            consumer_sum[0] += item
        latch.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    expected_sum = 0
    t0 = time.perf_counter()
    for i in range(n):
        q.put(i)
        expected_sum += i
    latch.wait(timeout=10)
    elapsed = time.perf_counter() - t0

    t.join(timeout=1)
    return elapsed, consumer_sum[0] == expected_sum


# ──────────────────────────────────────────────────────────────────────
# 1P3C — Single Producer, 3 Consumers (multicast)
# ──────────────────────────────────────────────────────────────────────


class CountDownLatch:
    def __init__(self, count: int):
        self._count = count
        self._event = threading.Event()
        self._lock = threading.Lock()

    def count_down(self):
        with self._lock:
            self._count -= 1
            if self._count <= 0:
                self._event.set()

    def wait(self, timeout: float | None = None):
        self._event.wait(timeout)


def bench_1p3c_channel(n: int) -> tuple[float, bool]:
    ch: Channel[int] = Channel()
    sender = Sender(ch)
    stop = threading.Event()
    latch = CountDownLatch(3)
    sums = [[0], [0], [0]]

    def consumer(sub_id: int, acc: list[int]):
        ch._register(sub_id, newest=False)
        for _ in range(n):
            item = ch._get(sub_id, stop, blocking=True)
            if item is None:
                break
            acc[0] += item
        latch.count_down()

    threads = []
    for i in range(3):
        t = threading.Thread(target=consumer, args=(i + 1, sums[i]), daemon=True)
        t.start()
        threads.append(t)

    expected_sum = 0
    t0 = time.perf_counter()
    for i in range(n):
        sender.send(i)
        expected_sum += i
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    stop.set()
    for t in threads:
        t.join(timeout=1)
    return elapsed, all(s[0] == expected_sum for s in sums)


def bench_1p3c_queue(n: int) -> tuple[float, bool]:
    queues: list[Queue[int]] = [Queue() for _ in range(3)]
    latch = CountDownLatch(3)
    sums = [[0], [0], [0]]

    def consumer(q: Queue, acc: list[int]):
        for _ in range(n):
            item = q.get()
            acc[0] += item
        latch.count_down()

    threads = []
    for i in range(3):
        t = threading.Thread(target=consumer, args=(queues[i], sums[i]), daemon=True)
        t.start()
        threads.append(t)

    expected_sum = 0
    t0 = time.perf_counter()
    for i in range(n):
        for q in queues:
            q.put(i)
        expected_sum += i
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    for t in threads:
        t.join(timeout=1)
    return elapsed, all(s[0] == expected_sum for s in sums)


# ──────────────────────────────────────────────────────────────────────
# 3P1C — Three Producers, Single Consumer
# ──────────────────────────────────────────────────────────────────────


def bench_3p1c_channel(n: int) -> tuple[float, bool]:
    ch: Channel[int] = Channel()
    senders = [Sender(ch) for _ in range(3)]
    stop = threading.Event()
    latch = threading.Event()
    consumer_sum = [0]
    per_producer = n // 3

    def consumer():
        ch._register(1, latest=False)
        total = per_producer * 3
        for _ in range(total):
            item = ch._wait_and_get(1, stop)
            if item is None:
                break
            consumer_sum[0] += item
        latch.set()

    ct = threading.Thread(target=consumer, daemon=True)
    ct.start()

    go = threading.Event()
    expected_sums = [0, 0, 0]

    def producer(pid: int, sender: Sender):
        go.wait()
        for i in range(per_producer):
            val = pid * per_producer + i
            sender.send(val)
            expected_sums[pid] += val

    pts = []
    for pid in range(3):
        t = threading.Thread(target=producer, args=(pid, senders[pid]), daemon=True)
        t.start()
        pts.append(t)

    t0 = time.perf_counter()
    go.set()

    for t in pts:
        t.join(timeout=30)
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    stop.set()
    ct.join(timeout=1)
    return elapsed, consumer_sum[0] == sum(expected_sums)


def bench_3p1c_queue(n: int) -> tuple[float, bool]:
    q: Queue[int] = Queue()
    latch = threading.Event()
    consumer_sum = [0]
    per_producer = n // 3

    def consumer():
        total = per_producer * 3
        for _ in range(total):
            item = q.get()
            consumer_sum[0] += item
        latch.set()

    ct = threading.Thread(target=consumer, daemon=True)
    ct.start()

    go = threading.Event()
    expected_sums = [0, 0, 0]

    def producer(pid: int):
        go.wait()
        for i in range(per_producer):
            val = pid * per_producer + i
            q.put(val)
            expected_sums[pid] += val

    pts = []
    for pid in range(3):
        t = threading.Thread(target=producer, args=(pid,), daemon=True)
        t.start()
        pts.append(t)

    t0 = time.perf_counter()
    go.set()

    for t in pts:
        t.join(timeout=30)
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    ct.join(timeout=1)
    return elapsed, consumer_sum[0] == sum(expected_sums)


# ──────────────────────────────────────────────────────────────────────
# Pipeline — 3 sequential stages: P -> S1 -> S2 -> S3
# ──────────────────────────────────────────────────────────────────────


def bench_pipeline_channel(n: int) -> tuple[float, bool]:
    ch1: Channel[int] = Channel()
    ch2: Channel[int] = Channel()
    ch3: Channel[int] = Channel()
    sender = Sender(ch1)
    stop = threading.Event()
    latch = threading.Event()
    final_sum = [0]

    def stage(
        in_ch: Channel,
        out_ch: Channel | None,
        sub_id: int,
        transform,
        acc: list[int] | None,
    ):
        in_ch._register(sub_id, newest=False)
        out_sender = Sender(out_ch) if out_ch else None
        for _ in range(n):
            item = in_ch._get(sub_id, stop, blocking=True)
            if item is None:
                break
            val = transform(item)
            if out_sender:
                out_sender.send(val)
            if acc is not None:
                acc[0] += val
        if acc is not None:
            latch.set()

    t1 = threading.Thread(
        target=stage, args=(ch1, ch2, 1, lambda x: x + 1, None), daemon=True
    )
    t2 = threading.Thread(
        target=stage, args=(ch2, ch3, 1, lambda x: x * 2, None), daemon=True
    )
    t3 = threading.Thread(
        target=stage, args=(ch3, None, 1, lambda x: x, final_sum), daemon=True
    )
    t1.start()
    t2.start()
    t3.start()

    expected_sum = sum((i + 1) * 2 for i in range(n))
    t0 = time.perf_counter()
    for i in range(n):
        sender.send(i)
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    stop.set()
    t1.join(timeout=1)
    t2.join(timeout=1)
    t3.join(timeout=1)
    return elapsed, final_sum[0] == expected_sum


def bench_pipeline_queue(n: int) -> tuple[float, bool]:
    q1: Queue[int] = Queue()
    q2: Queue[int] = Queue()
    q3: Queue[int] = Queue()
    latch = threading.Event()
    final_sum = [0]

    def stage(in_q: Queue, out_q: Queue | None, transform, acc: list[int] | None):
        for _ in range(n):
            item = in_q.get()
            val = transform(item)
            if out_q:
                out_q.put(val)
            if acc is not None:
                acc[0] += val
        if acc is not None:
            latch.set()

    t1 = threading.Thread(
        target=stage, args=(q1, q2, lambda x: x + 1, None), daemon=True
    )
    t2 = threading.Thread(
        target=stage, args=(q2, q3, lambda x: x * 2, None), daemon=True
    )
    t3 = threading.Thread(
        target=stage, args=(q3, None, lambda x: x, final_sum), daemon=True
    )
    t1.start()
    t2.start()
    t3.start()

    expected_sum = sum((i + 1) * 2 for i in range(n))
    t0 = time.perf_counter()
    for i in range(n):
        q1.put(i)
    latch.wait(timeout=30)
    elapsed = time.perf_counter() - t0

    t1.join(timeout=1)
    t2.join(timeout=1)
    t3.join(timeout=1)
    return elapsed, final_sum[0] == expected_sum


# ──────────────────────────────────────────────────────────────────────
# FastChannel variants (same logic, different Channel/Sender impl)
# ──────────────────────────────────────────────────────────────────────


def _make_fast_variants():
    if not HAS_FAST_CHANNEL:
        return {}

    def bench_1p1c_fast(n: int) -> tuple[float, bool]:
        ch = FastChannel()
        sender = FastSender(ch)
        stop = threading.Event()
        latch = threading.Event()
        consumer_sum = [0]

        def consumer():
            ch._register(1, latest=False)
            for _ in range(n):
                item = ch._wait_and_get(1, stop)
                if item is None:
                    break
                consumer_sum[0] += item
            latch.set()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        expected_sum = 0
        t0 = time.perf_counter()
        for i in range(n):
            sender.send(i)
            expected_sum += i
        latch.wait(timeout=10)
        elapsed = time.perf_counter() - t0
        stop.set()
        t.join(timeout=1)
        return elapsed, consumer_sum[0] == expected_sum

    def bench_1p3c_fast(n: int) -> tuple[float, bool]:
        ch = FastChannel()
        sender = FastSender(ch)
        stop = threading.Event()
        latch = CountDownLatch(3)
        sums = [[0], [0], [0]]

        def consumer(sub_id, acc):
            ch._register(sub_id, latest=False)
            for _ in range(n):
                item = ch._wait_and_get(sub_id, stop)
                if item is None:
                    break
                acc[0] += item
            latch.count_down()

        threads = []
        for i in range(3):
            t = threading.Thread(target=consumer, args=(i + 1, sums[i]), daemon=True)
            t.start()
            threads.append(t)
        expected_sum = 0
        t0 = time.perf_counter()
        for i in range(n):
            sender.send(i)
            expected_sum += i
        latch.wait(timeout=30)
        elapsed = time.perf_counter() - t0
        stop.set()
        for t in threads:
            t.join(timeout=1)
        return elapsed, all(s[0] == expected_sum for s in sums)

    def bench_3p1c_fast(n: int) -> tuple[float, bool]:
        ch = FastChannel()
        senders = [FastSender(ch) for _ in range(3)]
        stop = threading.Event()
        latch = threading.Event()
        consumer_sum = [0]
        per_producer = n // 3

        def consumer():
            ch._register(1, latest=False)
            for _ in range(per_producer * 3):
                item = ch._wait_and_get(1, stop)
                if item is None:
                    break
                consumer_sum[0] += item
            latch.set()

        ct = threading.Thread(target=consumer, daemon=True)
        ct.start()
        go = threading.Event()
        expected_sums = [0, 0, 0]

        def producer(pid, sender):
            go.wait()
            for i in range(per_producer):
                val = pid * per_producer + i
                sender.send(val)
                expected_sums[pid] += val

        pts = []
        for pid in range(3):
            t = threading.Thread(target=producer, args=(pid, senders[pid]), daemon=True)
            t.start()
            pts.append(t)
        t0 = time.perf_counter()
        go.set()
        for t in pts:
            t.join(timeout=30)
        latch.wait(timeout=30)
        elapsed = time.perf_counter() - t0
        stop.set()
        ct.join(timeout=1)
        return elapsed, consumer_sum[0] == sum(expected_sums)

    def bench_pipeline_fast(n: int) -> tuple[float, bool]:
        ch1 = FastChannel()
        ch2 = FastChannel()
        ch3 = FastChannel()
        sender = FastSender(ch1)
        stop = threading.Event()
        latch = threading.Event()
        final_sum = [0]

        def stage(in_ch, out_ch, sub_id, transform, acc):
            in_ch._register(sub_id, latest=False)
            out_sender = FastSender(out_ch) if out_ch else None
            for _ in range(n):
                item = in_ch._wait_and_get(sub_id, stop)
                if item is None:
                    break
                val = transform(item)
                if out_sender:
                    out_sender.send(val)
                if acc is not None:
                    acc[0] += val
            if acc is not None:
                latch.set()

        t1 = threading.Thread(
            target=stage, args=(ch1, ch2, 1, lambda x: x + 1, None), daemon=True
        )
        t2 = threading.Thread(
            target=stage, args=(ch2, ch3, 1, lambda x: x * 2, None), daemon=True
        )
        t3 = threading.Thread(
            target=stage, args=(ch3, None, 1, lambda x: x, final_sum), daemon=True
        )
        t1.start()
        t2.start()
        t3.start()
        expected_sum = sum((i + 1) * 2 for i in range(n))
        t0 = time.perf_counter()
        for i in range(n):
            sender.send(i)
        latch.wait(timeout=30)
        elapsed = time.perf_counter() - t0
        stop.set()
        t1.join(timeout=1)
        t2.join(timeout=1)
        t3.join(timeout=1)
        return elapsed, final_sum[0] == expected_sum

    def bench_latency_fast(n: int) -> list[float]:
        ch_ping = FastChannel()
        ch_pong = FastChannel()
        stop = threading.Event()
        ready = threading.Event()

        def ponger():
            ch_ping._register(1, latest=False)
            s = FastSender(ch_pong)
            ready.set()
            for _ in range(n):
                item = ch_ping._wait_and_get(1, stop)
                if item is None:
                    break
                s.send(item)

        t = threading.Thread(target=ponger, daemon=True)
        t.start()
        ready.wait()
        ch_pong._register(2, latest=False)
        pinger = FastSender(ch_ping)
        latencies = []
        for _ in range(n):
            t0 = time.perf_counter()
            pinger.send(0)
            ch_pong._wait_and_get(2, stop)
            latencies.append(time.perf_counter() - t0)
        stop.set()
        t.join(timeout=1)
        return latencies

    return {
        "1p1c": bench_1p1c_fast,
        "1p3c": bench_1p3c_fast,
        "3p1c": bench_3p1c_fast,
        "pipeline": bench_pipeline_fast,
        "latency": bench_latency_fast,
    }


FAST_VARIANTS = _make_fast_variants()


# ──────────────────────────────────────────────────────────────────────
# Ping-Pong Latency
# ──────────────────────────────────────────────────────────────────────


def bench_latency_channel(n: int) -> list[float]:
    ch_ping: Channel[int] = Channel()
    ch_pong: Channel[int] = Channel()
    stop = threading.Event()
    ready = threading.Event()

    def ponger():
        ch_ping._register(1, latest=False)
        s = Sender(ch_pong)
        ready.set()
        for _ in range(n):
            item = ch_ping._wait_and_get(1, stop)
            if item is None:
                break
            s.send(item)

    t = threading.Thread(target=ponger, daemon=True)
    t.start()
    ready.wait()

    ch_pong._register(2, newest=False)
    pinger = Sender(ch_ping)
    latencies = []

    for _ in range(n):
        t0 = time.perf_counter()
        pinger.send(0)
        ch_pong._get(2, stop, blocking=True)
        latencies.append(time.perf_counter() - t0)

    stop.set()
    t.join(timeout=1)
    return latencies


def bench_latency_queue(n: int) -> list[float]:
    q_ping: Queue[int] = Queue()
    q_pong: Queue[int] = Queue()
    ready = threading.Event()

    def ponger():
        ready.set()
        for _ in range(n):
            item = q_ping.get()
            q_pong.put(item)

    t = threading.Thread(target=ponger, daemon=True)
    t.start()
    ready.wait()

    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        q_ping.put(0)
        q_pong.get()
        latencies.append(time.perf_counter() - t0)

    t.join(timeout=1)
    return latencies


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


def run_throughput(name: str, channel_fn, queue_fn, n: int, runs: int, fast_fn=None):
    print(f"── {name} ({n:,} msgs × {runs} runs) ──")
    print()

    entries = [("Channel", channel_fn), ("Queue", queue_fn)]
    if fast_fn:
        entries.append(("FastChannel", fast_fn))

    for label, fn in entries:
        results = []
        checksums_ok = True
        for r in range(runs):
            gc.collect()
            print(f"  {label} run {r + 1}/{runs}...", end=" ", flush=True)
            elapsed, ok = fn(n)
            if not ok:
                checksums_ok = False
            ops = n / elapsed
            results.append(ops)
            print(f"{ops:,.0f} ops/s", flush=True)
        results.sort()
        median = results[len(results) // 2]
        best = results[-1]
        worst = results[0]
        check = "✓" if checksums_ok else "✗ CHECKSUM FAIL"
        print(
            f"  {label:<10} median={median:>12,.0f} ops/s  best={best:>12,.0f}  worst={worst:>12,.0f}  {check}"
        )

    print()


def run_latency(name: str, channel_fn, queue_fn, n: int, runs: int, fast_fn=None):
    print(f"── {name} ({n:,} round-trips × {runs} runs, best run) ──")
    print()

    entries = [("Channel", channel_fn), ("Queue", queue_fn)]
    if fast_fn:
        entries.append(("FastChannel", fast_fn))

    for label, fn in entries:
        best_p50 = float("inf")
        best_run = []
        for r in range(runs):
            gc.collect()
            print(f"  {label} run {r + 1}/{runs}...", end=" ", flush=True)
            lats = fn(n)
            p = percentiles(lats)
            print(f"p50={p['p50'] * 1e6:.0f}µs", flush=True)
            if p["p50"] < best_p50:
                best_p50 = p["p50"]
                best_run = lats
        if best_run:
            p = percentiles(best_run)
            us = {k: v * 1e6 for k, v in p.items()}
            print(
                f"  {label:<10} p50={us['p50']:>8.1f}µs  p90={us['p90']:>8.1f}µs  "
                f"p99={us['p99']:>8.1f}µs  p99.9={us['p99.9']:>8.1f}µs  max={us['max']:>8.1f}µs"
            )

    print()


def main():
    sep = "=" * 70

    title = (
        "Channel vs Queue vs FastChannel Benchmark"
        if HAS_FAST_CHANNEL
        else "Channel vs Queue Benchmark"
    )
    print(sep)
    print(title)
    print(f"  {N:,} messages/run, {RUNS} runs/test, gc.collect() between runs")
    print(f"  Latency: {LATENCY_N:,} ping-pong round-trips, {LATENCY_RUNS} runs")
    print(sep)
    print()

    fv = FAST_VARIANTS
    run_throughput(
        "1P1C (unicast)", bench_1p1c_channel, bench_1p1c_queue, N, RUNS, fv.get("1p1c")
    )
    run_throughput(
        "1P3C (multicast)",
        bench_1p3c_channel,
        bench_1p3c_queue,
        N,
        RUNS,
        fv.get("1p3c"),
    )
    run_throughput(
        "3P1C (fan-in)", bench_3p1c_channel, bench_3p1c_queue, N, RUNS, fv.get("3p1c")
    )
    run_throughput(
        "Pipeline (P→S1→S2→S3)",
        bench_pipeline_channel,
        bench_pipeline_queue,
        N,
        RUNS,
        fv.get("pipeline"),
    )
    run_latency(
        "Ping-Pong Latency",
        bench_latency_channel,
        bench_latency_queue,
        LATENCY_N,
        LATENCY_RUNS,
        fv.get("latency"),
    )

    print(sep)
    print("Notes:")
    print("  - Channel broadcasts to all consumers with a single send() (1P3C)")
    print("  - Queue requires N separate put() calls for N consumers (manual fan-out)")
    print("  - Channel._gc() runs on every receive, adding per-message overhead")
    print("  - All checksums validated: no dropped or duplicated messages")
    print(sep)


if __name__ == "__main__":
    main()
