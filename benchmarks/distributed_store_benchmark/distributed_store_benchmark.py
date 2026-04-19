#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Mooncake KVCache Distributed Store Benchmark Tool

Drop-in analog of storage_benchmark.py, but instead of writing blocks to a
local file via pread/pwrite, this benchmark exercises the Mooncake distributed
object store (MooncakeDistributedStore) configured via a config.json file.

Each hash_id from the trace becomes an object key. Existing keys are fetched
(read path / prefix reuse), missing keys are written (put path).
"""

import argparse
import json
import os
import re
import socket
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from mooncake.store import MooncakeDistributedStore

# ============================================================================
# Constants
# ============================================================================

BLOCK_SIZE_TOKENS = 512
DEFAULT_BYTES_PER_TOKEN = 2048  # 7B model FP16 (2KB per token)
MIN_LATENCY_MS = 0.001

# Model KVCache sizes (bytes per token)
# Source: https://lmcache.ai/kv_cache_calculator.html
MODEL_BYTES_PER_TOKEN = {
    "llama-3.1-405b": 327680,
    "qwen3-32b": 81920,
    "deepseek-v3": 1748992,
    "glm-4.6": 157013,
    "default": DEFAULT_BYTES_PER_TOKEN,
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KVCacheRequest:
    timestamp: float
    hash_ids: List[int]
    input_length: int
    output_length: int


# ============================================================================
# Config Loading
# ============================================================================

_SIZE_UNITS = {
    'B': 1,
    'K': 1024, 'KB': 1024, 'KIB': 1024,
    'M': 1024 ** 2, 'MB': 1024 ** 2, 'MIB': 1024 ** 2,
    'G': 1024 ** 3, 'GB': 1024 ** 3, 'GIB': 1024 ** 3,
    'T': 1024 ** 4, 'TB': 1024 ** 4, 'TIB': 1024 ** 4,
}


def parse_size(value) -> int:
    """Parse a size value like '25GB', '4GB', '512MB', or an int into bytes."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip().upper().replace(' ', '')
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([A-Z]*)', s)
    if not m:
        raise ValueError(f"Invalid size: {value!r}")
    num, unit = m.group(1), m.group(2) or 'B'
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown size unit: {unit!r} in {value!r}")
    return int(float(num) * _SIZE_UNITS[unit])


def load_config(config_path: str) -> Dict:
    """Load and validate config.json."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    required = ['metadata_server', 'master_server_address',
                'global_segment_size', 'local_buffer_size',
                'protocol', 'device_name']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg


# ============================================================================
# Distributed Store Wrapper
# ============================================================================

class DistributedStoreBackend:
    """Mooncake distributed store backend with per-op latency tracking."""

    def __init__(self, config: Dict, bytes_per_token: int,
                 block_size_tokens: int, local_hostname: Optional[str] = None,
                 key_prefix: str = 'bench'):
        self.block_size_tokens = block_size_tokens
        self.bytes_per_token = bytes_per_token
        self.block_size_bytes = block_size_tokens * bytes_per_token
        self.key_prefix = key_prefix

        # Pre-allocated data buffer — repeating byte pattern (not all zeros) so
        # any on-wire or on-disk compression does not distort measurements.
        pattern = bytes([(i & 0xFF) for i in range(256)])
        repeats = (self.block_size_bytes // len(pattern)) + 1
        self._data_buffer = (pattern * repeats)[:self.block_size_bytes]

        host = (local_hostname
                or os.getenv('LOCAL_HOSTNAME')
                or socket.gethostname()
                or 'localhost')

        global_segment_size = parse_size(config['global_segment_size'])
        local_buffer_size = parse_size(config['local_buffer_size'])

        self.store = MooncakeDistributedStore()
        retcode = self.store.setup(
            host,
            config['metadata_server'],
            global_segment_size,
            local_buffer_size,
            config['protocol'],
            config['device_name'],
            config['master_server_address'],
        )
        if retcode:
            raise RuntimeError(
                f"MooncakeDistributedStore.setup failed with retcode={retcode}"
            )

        # Track keys we've successfully written so we can clean up at the end
        # (and so a single benchmark run is idempotent even if the cluster is
        # shared).
        self._written_keys: set = set()

        self.stats = {
            'read_count': 0,
            'write_count': 0,
            'read_bytes': 0,
            'write_bytes': 0,
            'read_latencies_ms': [],
            'write_latencies_ms': [],
        }

    def _key(self, hash_id: int) -> str:
        return f"{self.key_prefix}:{self.block_size_tokens}:{hash_id}"

    def block_exists(self, hash_id: int) -> bool:
        # Local cache check first — avoids a round-trip for keys this client
        # already wrote in this run. For cross-run prefix reuse, the store
        # itself is authoritative.
        if hash_id in self._written_keys:
            return True
        try:
            return bool(self.store.is_exist(self._key(hash_id)))
        except Exception:
            return False

    def read_block(self, hash_id: int) -> float:
        key = self._key(hash_id)
        start = time.perf_counter()
        data = self.store.get(key)
        latency_ms = (time.perf_counter() - start) * 1000.0
        n = len(data) if data else 0
        if n == 0:
            # Miss — don't pollute latency stats with a failed fetch
            return 0.0
        self.stats['read_count'] += 1
        self.stats['read_bytes'] += n
        self.stats['read_latencies_ms'].append(latency_ms)
        return latency_ms

    def write_block(self, hash_id: int) -> float:
        key = self._key(hash_id)
        data = self._data_buffer
        start = time.perf_counter()
        retcode = self.store.put(key, data)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if retcode != 0:
            print(f"Warning: put failed for key {key} (retcode={retcode})")
            return 0.0
        self._written_keys.add(hash_id)
        self.stats['write_count'] += 1
        self.stats['write_bytes'] += len(data)
        self.stats['write_latencies_ms'].append(latency_ms)
        return latency_ms

    def cleanup(self):
        """Best-effort removal of keys written by this run."""
        removed = 0
        for hash_id in list(self._written_keys):
            try:
                if self.store.remove(self._key(hash_id)) == 0:
                    removed += 1
            except Exception:
                pass
        self._written_keys.clear()
        return removed

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass

    def get_stats(self) -> Dict:
        def calc(latencies):
            if not latencies:
                return {'avg_ms': 0, 'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0}
            return {'avg_ms': statistics.mean(latencies),
                    **calc_percentiles(latencies)}

        return {
            'read': {
                'count': self.stats['read_count'],
                'mb': self.stats['read_bytes'] / 1024 / 1024,
                **calc(self.stats['read_latencies_ms']),
            },
            'write': {
                'count': self.stats['write_count'],
                'mb': self.stats['write_bytes'] / 1024 / 1024,
                **calc(self.stats['write_latencies_ms']),
            },
            'total_blocks': len(self._written_keys),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# Benchmark Layer
# ============================================================================

class StoreBenchmark:
    """KVCache benchmark driving Mooncake distributed store."""

    def __init__(self, backend: DistributedStoreBackend):
        self.backend = backend
        self.block_size_tokens = backend.block_size_tokens
        self.stats = {
            'total_requests': 0,
            'total_blocks': 0,
            'read_blocks': 0,
            'write_blocks': 0,
            'prefix_hit_blocks': 0,
            'request_latencies_ms': [],
        }

    def process_request(self, req: KVCacheRequest) -> float:
        self.stats['total_requests'] += 1
        self.stats['total_blocks'] += len(req.hash_ids)
        total_latency = 0.0

        for hash_id in req.hash_ids:
            if self.backend.block_exists(hash_id):
                total_latency += self.backend.read_block(hash_id)
                self.stats['read_blocks'] += 1
                self.stats['prefix_hit_blocks'] += 1
            else:
                total_latency += self.backend.write_block(hash_id)
                self.stats['write_blocks'] += 1

        latency_ms = total_latency if total_latency > 0 else MIN_LATENCY_MS
        self.stats['request_latencies_ms'].append(latency_ms)
        return latency_ms

    def get_stats(self) -> Dict:
        storage_stats = self.backend.get_stats()
        rl = self.stats['request_latencies_ms']
        if rl:
            latency_stats = {'avg_ms': statistics.mean(rl),
                             **calc_percentiles(rl)}
        else:
            latency_stats = {'avg_ms': 0, 'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0}

        total = self.stats['total_blocks']
        reads = self.stats['read_blocks']
        writes = self.stats['write_blocks']
        return {
            'total_requests': self.stats['total_requests'],
            'total_blocks': total,
            'read_blocks': reads,
            'write_blocks': writes,
            'prefix_hit_blocks': self.stats['prefix_hit_blocks'],
            'block_hit_rate': reads / total if total else 0,
            'write_ratio': writes / total if total else 0,
            'tokens_per_block': self.block_size_tokens,
            'latency': latency_stats,
            'storage': storage_stats,
        }


# ============================================================================
# Utility
# ============================================================================

def calc_percentiles(data: List[float]) -> Dict[str, float]:
    if not data:
        return {'p50_ms': 0, 'p95_ms': 0, 'p99_ms': 0}
    sorted_data = sorted(data)
    n = len(sorted_data)

    def pct(p):
        index = (n - 1) * p / 100
        lower = int(index)
        upper = min(lower + 1, n - 1)
        if lower == upper:
            return sorted_data[lower]
        w = index - lower
        return sorted_data[lower] * (1 - w) + sorted_data[upper] * w

    return {'p50_ms': pct(50), 'p95_ms': pct(95), 'p99_ms': pct(99)}


# ============================================================================
# Trace Loader (same format as storage_benchmark.py)
# ============================================================================

class TraceLoader:
    def __init__(self, trace_path: str):
        self.trace_path = trace_path
        self.requests: List[KVCacheRequest] = []
        self._load()

    def _load(self):
        line_num = 0
        with open(self.trace_path, 'r') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    if not all(k in req for k in
                               ['timestamp', 'hash_ids',
                                'input_length', 'output_length']):
                        print(f"Warning: Line {line_num} missing fields, skipping")
                        continue
                    if not isinstance(req['hash_ids'], list):
                        print(f"Warning: Line {line_num} invalid hash_ids, skipping")
                        continue
                    self.requests.append(KVCacheRequest(
                        timestamp=float(req['timestamp']),
                        hash_ids=req['hash_ids'],
                        input_length=int(req['input_length']),
                        output_length=int(req['output_length']),
                    ))
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Warning: Line {line_num} invalid: {e}, skipping")

    def get_requests(self) -> List[KVCacheRequest]:
        return self.requests


# ============================================================================
# Runner
# ============================================================================

def run_benchmark(trace_path: str, config: Dict, bytes_per_token: int,
                  max_requests: Optional[int], block_size_tokens: int,
                  replay_timestamps: bool, time_scale: float,
                  key_prefix: str, cleanup: bool) -> Dict:
    block_size_bytes = block_size_tokens * bytes_per_token

    print(f"\n{'='*80}")
    print(f"Running: {Path(trace_path).name}")
    print(f"Backend: Mooncake Distributed Store")
    print(f"  metadata_server:       {config['metadata_server']}")
    print(f"  master_server_address: {config['master_server_address']}")
    print(f"  protocol:              {config['protocol']}")
    print(f"  device_name:           {config['device_name']}")
    print(f"  global_segment_size:   {config['global_segment_size']} "
          f"({parse_size(config['global_segment_size']):,} bytes)")
    print(f"  local_buffer_size:     {config['local_buffer_size']} "
          f"({parse_size(config['local_buffer_size']):,} bytes)")
    print(f"Block size: {block_size_tokens} tokens/block "
          f"({block_size_bytes:,} bytes)")
    print(f"Bytes per token: {bytes_per_token}")
    print(f"Key prefix: {key_prefix}")
    print(f"Timestamp replay: "
          f"{'Enabled' if replay_timestamps else 'Disabled'}")
    if replay_timestamps:
        print(f"Time scale: {time_scale}x")
    print(f"{'='*80}")

    requests = TraceLoader(trace_path).get_requests()
    if max_requests:
        requests = requests[:max_requests]
    print(f"Loaded {len(requests)} requests")

    with DistributedStoreBackend(
        config, bytes_per_token, block_size_tokens,
        key_prefix=key_prefix,
    ) as backend:
        benchmark = StoreBenchmark(backend)

        start_time = time.perf_counter()
        total_io_time = 0.0
        last_timestamp = None
        base_time = time.time()

        for i, req in enumerate(requests):
            if replay_timestamps and last_timestamp is not None:
                delta_ms = req.timestamp - last_timestamp
                sleep_time = delta_ms / 1000.0 / time_scale
                if sleep_time > 0:
                    time.sleep(sleep_time)

            req_start = time.perf_counter()
            benchmark.process_request(req)
            total_io_time += time.perf_counter() - req_start
            last_timestamp = req.timestamp

            if (i + 1) % 100 == 0:
                if replay_timestamps:
                    elapsed = time.time() - base_time
                    sim = (req.timestamp - requests[0].timestamp) / 1000.0 / time_scale
                    print(f"  Processed {i + 1}/{len(requests)}... "
                          f"(wall: {elapsed:.1f}s, sim: {sim:.1f}s, "
                          f"io: {total_io_time:.1f}s)")
                else:
                    print(f"  Processed {i + 1}/{len(requests)}...")

        elapsed = time.perf_counter() - start_time
        stats = benchmark.get_stats()

        if cleanup:
            removed = backend.cleanup()
            print(f"Cleanup: removed {removed} keys")

    io_time = total_io_time if replay_timestamps else elapsed
    return {
        'trace_file': Path(trace_path).name,
        'total_requests': len(requests),
        'simulation_time_s': elapsed,
        'io_time_s': io_time,
        'wall_time_s': elapsed,
        'requests_per_second': len(requests) / io_time if io_time > 0 else 0,
        'timestamp_replay_enabled': replay_timestamps,
        'time_scale': time_scale,
        'bytes_per_token': bytes_per_token,
        'block_size_tokens': block_size_tokens,
        **stats,
    }


def print_results(results: List[Dict]):
    for i, r in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"  [{i}/{len(results)}] {r['trace_file']}")
        print(f"{'='*80}")

        print(f"\n[Performance Overview]")
        print(f"  Total Requests:           {r['total_requests']:,}")
        print(f"  Queries Per Second (QPS): {r['requests_per_second']:.2f}")
        print(f"  Cache Hit Rate:           {r['block_hit_rate']:.2%}")
        print(f"  Write Ratio:              {r['write_ratio']:.2%}")
        print(f"  Total Blocks:             {r['total_blocks']:,}")
        print(f"    Read Blocks:            {r['read_blocks']:,}")
        print(f"    Write Blocks:           {r['write_blocks']:,}")
        print(f"    Prefix Hits:            {r['prefix_hit_blocks']:,}")

        print(f"\n[Latency Analysis]")
        rl = r['latency']
        print(f"  Request Latency: Avg={rl['avg_ms']:.2f}ms, "
              f"P50={rl['p50_ms']:.2f}ms, P95={rl['p95_ms']:.2f}ms, "
              f"P99={rl['p99_ms']:.2f}ms")
        rd = r['storage']['read']
        wr = r['storage']['write']
        print(f"  Single I/O Operation (Per Block):")
        print(f"    Read:  Avg={rd.get('avg_ms', 0):.3f}ms, "
              f"P50={rd.get('p50_ms', 0):.3f}ms, "
              f"P95={rd.get('p95_ms', 0):.3f}ms, "
              f"P99={rd.get('p99_ms', 0):.3f}ms")
        print(f"    Write: Avg={wr.get('avg_ms', 0):.3f}ms, "
              f"P50={wr.get('p50_ms', 0):.3f}ms, "
              f"P95={wr.get('p95_ms', 0):.3f}ms, "
              f"P99={wr.get('p99_ms', 0):.3f}ms")

        print(f"\n[I/O & Bandwidth]")
        print(f"  Total Read I/O:  {r['storage']['read']['mb']:>10.1f} MB  "
              f"({r['storage']['read']['count']:,} ops)")
        print(f"  Total Write I/O: {r['storage']['write']['mb']:>10.1f} MB  "
              f"({r['storage']['write']['count']:,} ops)")

        rd = r['storage']['read']
        wr = r['storage']['write']
        # Per-op bandwidth = bytes / sum(op latencies). sum = count * avg.
        read_time_s = rd['count'] * rd.get('avg_ms', 0) / 1000
        write_time_s = wr['count'] * wr.get('avg_ms', 0) / 1000
        if read_time_s > 0:
            print(f"  Read Bandwidth:      "
                  f"{rd['mb'] / read_time_s:>10.1f} MB/s  "
                  f"(per-op: {rd['mb']:.1f} MB / {read_time_s:.2f}s)")
        if write_time_s > 0:
            print(f"  Write Bandwidth:     "
                  f"{wr['mb'] / write_time_s:>10.1f} MB/s  "
                  f"(per-op: {wr['mb']:.1f} MB / {write_time_s:.2f}s)")
        io_time = r['io_time_s']
        if io_time > 0:
            bw = (rd['mb'] + wr['mb']) / io_time
            print(f"  Aggregate Bandwidth: {bw:>10.1f} MB/s  "
                  f"(over {io_time:.2f}s wall I/O time)")

        print(f"\n[Execution Time]")
        if r.get('timestamp_replay_enabled'):
            print(f"  Wall Time (Total):   {r['wall_time_s']:>10.2f} s")
            print(f"  I/O Time (Actual):   {r['io_time_s']:>10.2f} s")
            print(f"  Sleep Time (Replay): "
                  f"{r['wall_time_s'] - r['io_time_s']:>10.2f} s")
        else:
            print(f"  Total Execution Time: {r['wall_time_s']:>10.2f} s")

    print(f"\n{'='*80}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mooncake KVCache Distributed Store Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test against the default config.json (100 requests)
  python distributed_store_benchmark.py --config=config.json \\
      --scenario=toolagent --max-requests=100

  # All scenarios with a large-model preset
  python distributed_store_benchmark.py --config=config.json \\
      --scenario=all --model=llama-3.1-405b

  # Realistic replay (10x speed)
  python distributed_store_benchmark.py --config=config.json \\
      --scenario=toolagent --replay-timestamps --time-scale=0.1

Config file (config.json) must contain:
  metadata_server, master_server_address, global_segment_size,
  local_buffer_size, protocol, device_name
Sizes may be given as strings like "25GB", "4GB", "512MB".
        """,
    )

    default_config = Path(__file__).parent / 'config.json'
    default_trace = Path(__file__).parent / '..' / '..' / 'FAST25-release' / 'traces'

    parser.add_argument('--config', type=str, default=str(default_config),
                        help='Path to Mooncake config.json')
    parser.add_argument('--trace-dir', type=str, default=str(default_trace),
                        help='Trace files directory')
    parser.add_argument('--scenario', type=str,
                        choices=['conversation', 'synthetic', 'toolagent', 'all'],
                        default='toolagent', help='Test scenario')
    parser.add_argument('--model', type=str,
                        choices=list(MODEL_BYTES_PER_TOKEN.keys()),
                        default='default',
                        help='Model preset (overrides --bytes-per-token)')
    parser.add_argument('--bytes-per-token', type=int,
                        default=DEFAULT_BYTES_PER_TOKEN,
                        help=f'Bytes per token (default {DEFAULT_BYTES_PER_TOKEN})')
    parser.add_argument('--max-requests', type=int, default=None,
                        help='Maximum number of requests (default: all)')
    parser.add_argument('--block-size-tokens', type=int, default=BLOCK_SIZE_TOKENS,
                        help=f'Number of tokens per block (default {BLOCK_SIZE_TOKENS})')
    parser.add_argument('--replay-timestamps', action='store_true',
                        help='Replay trace timestamps')
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='Time scaling factor for timestamp replay')
    parser.add_argument('--key-prefix', type=str, default='bench',
                        help='Prefix applied to every object key')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip removing keys written during the run')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"{'Mooncake KVCache Distributed Store Benchmark':^80}")
    print(f"{'='*80}")

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    bytes_per_token = MODEL_BYTES_PER_TOKEN.get(args.model, args.bytes_per_token)
    if args.model != 'default':
        print(f"Using model preset: {args.model} "
              f"({bytes_per_token} bytes/token, ~{bytes_per_token/1024:.1f} KB/token)")
    else:
        print(f"Using custom bytes_per_token: {bytes_per_token}")

    # Use a per-run key prefix so concurrent clients don't collide.
    key_prefix = f"{args.key_prefix}:{os.getpid()}"

    scenarios = (['conversation', 'synthetic', 'toolagent']
                 if args.scenario == 'all' else [args.scenario])
    trace_files = {
        'conversation': 'conversation_trace.jsonl',
        'synthetic': 'synthetic_trace.jsonl',
        'toolagent': 'toolagent_trace.jsonl',
    }

    results = []
    for scenario in scenarios:
        trace_path = Path(args.trace_dir) / trace_files[scenario]
        if not trace_path.exists():
            print(f"Warning: Trace file not found: {trace_path}")
            continue
        results.append(run_benchmark(
            str(trace_path), config, bytes_per_token,
            args.max_requests, args.block_size_tokens,
            args.replay_timestamps, args.time_scale,
            f"{key_prefix}:{scenario}",
            cleanup=not args.no_cleanup,
        ))

    if results:
        print_results(results)


if __name__ == '__main__':
    main()
